import pygame
import cv2
import numpy as np
from threading import Thread, Event, Lock
from queue import Queue, Empty, Full
import time
import rtmidi
import sys
import os
import subprocess
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from pygame.locals import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration data class"""
    fps: int = 60
    buffer_size: int = 5
    midi_channel: int = 0
    sync_interval: int = 30
    max_frame_skip: int = 5
    effects: Dict[str, Dict[str, Any]] = None
    
    @classmethod
    def load(cls, config_path: str = 'config.json') -> 'Config':
        """Load configuration from file"""
        default_effects = {
            'pad_1': {'type': 'color', 'color': (255, 255, 255)},
            'pad_2': {'type': 'color', 'color': (0, 0, 0)},
            'pad_3': {'type': 'image', 'path': 'overlay1.png'},
            'pad_4': {'type': 'image', 'path': 'overlay2.png'}
        }
        
        config_data = {
            'fps': 60,
            'buffer_size': 5,
            'midi_channel': 0,
            'sync_interval': 30,
            'max_frame_skip': 5,
            'effects': default_effects
        }
        
        try:
            with open(config_path, 'r') as f:
                config_data.update(json.load(f))
        except FileNotFoundError:
            logger.warning("No config file found, using defaults")
        except json.JSONDecodeError:
            logger.error("Invalid config file, using defaults")
        
        return cls(**config_data)

class Effect:
    """Base class for visual effects"""
    def __init__(self, intensity: float = 1.0):
        self.intensity = max(0.0, min(1.0, intensity))
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ColorEffect(Effect):
    """Color overlay effect"""
    def __init__(self, color: Tuple[int, int, int], intensity: float = 1.0):
        super().__init__(intensity)
        self.color = color
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(
            frame, 1 - self.intensity,
            np.full_like(frame, self.color),
            self.intensity, 0
        )

class ImageOverlayEffect(Effect):
    """Image overlay effect with alpha blending"""
    def __init__(self, overlay_image: np.ndarray, overlay_alpha: np.ndarray, intensity: float = 1.0):
        super().__init__(intensity)
        self.overlay_image = overlay_image
        self.overlay_alpha = overlay_alpha
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        alpha = self.overlay_alpha * self.intensity
        alpha_3d = np.stack([alpha] * 3, axis=-1)
        return (frame * (1 - alpha_3d) + self.overlay_image * alpha_3d).astype(np.uint8)

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    def __init__(self, log_interval: int = 100):
        self.frame_times: List[float] = []
        self.midi_latency: List[float] = []
        self.buffer_usage: List[int] = []
        self.log_interval = log_interval
        self.last_frame_time = time.time()
    
    def update_frame_time(self):
        current_time = time.time()
        frame_time = (current_time - self.last_frame_time) * 1000
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        self._check_logging()
    
    def update_midi_latency(self, latency: float):
        self.midi_latency.append(latency)
        self._check_logging()
    
    def update_buffer_usage(self, usage: int):
        self.buffer_usage.append(usage)
    
    def _check_logging(self):
        if len(self.frame_times) >= self.log_interval:
            self._log_metrics()
    
    def _log_metrics(self):
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            avg_midi_latency = np.mean(self.midi_latency) if self.midi_latency else 0
            avg_buffer_usage = np.mean(self.buffer_usage) if self.buffer_usage else 0
            
            logger.info(
                f"Performance Metrics:\n"
                f"Average frame time: {avg_frame_time:.2f}ms\n"
                f"Average MIDI latency: {avg_midi_latency:.2f}ms\n"
                f"Average buffer usage: {avg_buffer_usage:.1f}%"
            )
            
            self.frame_times.clear()
            self.midi_latency.clear()
            self.buffer_usage.clear()

class MidiVDJ:
    def __init__(self, video_path: Optional[str] = None, midi_port: int = 0,
                 test_mode: bool = False, display_index: int = 0):
        # Remove custom_resolution parameter
        """Initialize the MIDI VDJ application"""
        # Initialize control variables first
        self.volume = 1.0
        self.running = True
        self.test_mode = test_mode
        self._image_cache = {}
        
        # Load config and create performance monitor
        self.config = Config.load()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize threading controls
        self.shutdown_event = Event()
        self.effects_lock = Lock()
        self.buffer_lock = Lock()
        self.frame_queue = Queue(maxsize=10)
        self.processed_frame_queue = Queue(maxsize=5)
        
        # Initialize video first to get dimensions
        self._init_video(video_path)
        
        # Initialize display with video dimensions
        self._init_display(display_index)
        
        # Initialize MIDI
        self._init_midi(midi_port)
        
        # Initialize audio after video
        if not test_mode:
            self._init_audio(video_path)
        
        # Initialize effects with correct dimensions
        self._init_effects()
        
        # Initialize keyboard mapping
        self._init_keyboard_mapping()
        
        # Initialize remaining state
        self.active_effects: Dict[int, Effect] = {}
        self.frame_buffer: List[np.ndarray] = []
        self.active_keys = set()
        self.last_frame_time = time.time()

    def _init_keyboard_mapping(self):
        """Initialize keyboard controls"""
        self.key_to_note_mapping = {
            K_1: 0,  K_2: 1,  K_3: 2,  K_4: 3,   # Row 1: 1-4
            K_q: 4,  K_w: 5,  K_e: 6,  K_r: 7,   # Row 2: Q-R
            K_a: 8,  K_s: 9,  K_d: 10, K_f: 11,  # Row 3: A-F
            K_z: 12, K_x: 13, K_c: 14, K_v: 15   # Row 4: Z-V
        }

    def _init_video(self, video_path: Optional[str]):
        """Initialize video capture and settings"""
        if not self.test_mode:
            try:
                self.cap = cv2.VideoCapture(video_path)
                if not self.cap.isOpened():
                    raise ValueError(f"Could not open video file: {video_path}")
                
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.frame_time = 1.0 / self.fps
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Set OpenCV buffer size
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                logger.info(f"Video loaded: {self.width}x{self.height} @ {self.fps}fps")
            except Exception as e:
                raise RuntimeError(f"Video initialization failed: {e}")
        else:
            self.width = 1920
            self.height = 1080
            self.fps = 60
            self.frame_time = 1.0 / self.fps

    def _init_display(self, display_index: int):
        """Initialize pygame display with video resolution"""
        try:
            pygame.init()
            self.display_info = pygame.display.get_desktop_sizes()
            
            if display_index >= len(self.display_info):
                logger.warning(f"Display {display_index} not available, using 0")
                self.display_index = 0
            else:
                self.display_index = display_index
            
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('MPD218 VDJ' + (' - Test Mode' if self.test_mode else ''))
            
        except pygame.error as e:
            raise RuntimeError(f"Display initialization failed: {e}")

    def _init_midi(self, midi_port: int):
        """Initialize MIDI interface"""
        try:
            self.midi_in = rtmidi.MidiIn()
            available_ports = self.midi_in.get_ports()
            
            if not available_ports:
                raise ValueError("No MIDI ports available")
            
            self.midi_in.open_port(midi_port)
            logger.info(f"Connected to MIDI port {midi_port}: {available_ports[midi_port]}")
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"MIDI initialization failed: {e}")

    def _init_audio(self, video_path: str):
        """Initialize audio system"""
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            self._extract_and_load_audio(video_path)
        except Exception as e:
            logger.error(f"Audio initialization failed: {e}")

    def _extract_and_load_audio(self, video_path: str) -> bool:
        """Extract and load audio from video"""
        audio_path = "audio.wav"
        try:
            subprocess.call([
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '2', audio_path, '-y'
            ])
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play(-1)
            pygame.mixer.music.set_volume(self.volume)
            return True
        except Exception as e:
            logger.error(f"Audio extraction/loading failed: {e}")
            return False

    def _init_effects(self):
        """Initialize visual effects"""
        try:
            self.overlay_image1_bgr, self.overlay_image1_alpha = self._load_image('overlay1.png')
            logger.info("Loaded overlay1.png")
        except Exception as e:
            logger.error(f"Failed to load overlay1.png: {e}")
            self.overlay_image1_bgr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.overlay_image1_alpha = np.zeros((self.height, self.width), dtype=float)

        try:
            self.overlay_image2_bgr, self.overlay_image2_alpha = self._load_image('overlay2.png')
            logger.info("Loaded overlay2.png")
        except Exception as e:
            logger.error(f"Failed to load overlay2.png: {e}")
            self.overlay_image2_bgr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.overlay_image2_alpha = np.zeros((self.height, self.width), dtype=float)

    def _load_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process overlay image with caching"""
        if image_path in self._image_cache:
            return self._image_cache[image_path]

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # No need to check dimensions - always scale to video resolution
        if image.shape[2] == 4:
            # First, flip horizontally
            image = cv2.flip(image, 1)
            
            # Resize to match video dimensions exactly
            bgr = cv2.resize(image[:, :, :3], (self.width, self.height))
            alpha = cv2.resize(image[:, :, 3], (self.width, self.height))
            
            # Convert alpha to float and normalize
            alpha = alpha.astype(float) / 255
        else:
            # For images without alpha channel
            image = cv2.flip(image, 1)  # Flip horizontally
            bgr = cv2.resize(image, (self.width, self.height))
            alpha = np.ones((self.height, self.width), dtype=float)

        result = (bgr, alpha)
        self._image_cache[image_path] = result
        return result


    def _process_frames_thread(self):
        """Frame processing thread"""
        while not self.shutdown_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                processed_frame = self._apply_effects(frame)
                self.processed_frame_queue.put(processed_frame)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Frame processing error: {e}")

    def _handle_events(self) -> bool:
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self._toggle_fullscreen()
                elif event.key in self.key_to_note_mapping and event.key not in self.active_keys:
                    note = self.key_to_note_mapping[event.key]
                    self._trigger_effect(note)
                    self.active_keys.add(event.key)
            
            elif event.type == pygame.KEYUP:
                if event.key in self.key_to_note_mapping:
                    note = self.key_to_note_mapping[event.key]
                    self._release_effect(note)
                    self.active_keys.discard(event.key)
        
        return True

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        is_fullscreen = bool(pygame.display.get_surface().get_flags() & pygame.FULLSCREEN)
        if is_fullscreen:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{self.display_info[self.display_index][0]},{0}"
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    def _handle_volume_control(self, cc_value: int):
        """Handle volume control from MIDI"""
        if cc_value == 1:
            self.volume = min(1.0, self.volume + 0.01)
        elif cc_value == 127:
            self.volume = max(0.0, self.volume - 0.01)
        
        if pygame.mixer.get_init():
            pygame.mixer.music.set_volume(self.volume)

    def _trigger_effect(self, note: int):
        """Trigger visual effect"""
        with self.effects_lock:
            if 0 <= note < 16:
                if note < 4:
                    self.active_effects[note] = ColorEffect((255, 255, 255))
                elif note < 8:
                    self.active_effects[note] = ColorEffect((0, 0, 0))
                elif note < 12:
                    self.active_effects[note] = ImageOverlayEffect(
                        self.overlay_image1_bgr,
                        self.overlay_image1_alpha
                    )
                else:
                    self.active_effects[note] = ImageOverlayEffect(
                        self.overlay_image2_bgr,
                        self.overlay_image2_alpha
                    )

    def _release_effect(self, note: int):
        """Release active effect"""
        with self.effects_lock:
            self.active_effects.pop(note, None)

    def _apply_effects(self, frame: np.ndarray) -> np.ndarray:
        """Apply visual effects with optimization"""
        with self.effects_lock:
            if not self.active_effects:
                return frame

            result = frame.copy()
            
            # Group effects by type for batch processing
            color_effects = []
            image_effects = []
            
            for effect in self.active_effects.values():
                if isinstance(effect, ColorEffect):
                    color_effects.append(effect)
                elif isinstance(effect, ImageOverlayEffect):
                    image_effects.append(effect)
            
            # Batch process color effects
            if color_effects:
                combined_color = np.mean([effect.color for effect in color_effects], axis=0)
                combined_intensity = np.mean([effect.intensity for effect in color_effects])
                result = cv2.addWeighted(
                    result, 1 - combined_intensity,
                    np.full_like(result, combined_color),
                    combined_intensity, 0
                )
            
            # Apply image effects
            for effect in image_effects:
                result = effect.apply(result)
                
            return result

    def handle_midi(self):
        """MIDI event handling thread"""
        logger.info("MIDI handling thread started")
        while not self.shutdown_event.is_set():
            try:
                msg = self.midi_in.get_message()
                if msg:
                    message, delta_time = msg
                    self.performance_monitor.update_midi_latency(delta_time * 1000)
                    
                    if len(message) >= 3:
                        status, note, velocity = message
                        
                        if status == 180 and note == 4:
                            self._handle_volume_control(velocity)
                        elif status in [148, 144] and velocity > 0:
                            self._trigger_effect(note)
                        elif status in [132, 128] or (status in [148, 144] and velocity == 0):
                            self._release_effect(note)
                
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"MIDI handling error: {e}")
                break

    def run(self):
        """Main application loop with strict audio sync"""
        try:
            # Start threads
            midi_thread = Thread(target=self.handle_midi)
            midi_thread.daemon = True
            midi_thread.start()

            process_thread = Thread(target=self._process_frames_thread)
            process_thread.daemon = True
            process_thread.start()
            
            clock = pygame.time.Clock()
            frame_count = 0
            frame_time_ms = 1000 / self.fps  # Time per frame in milliseconds
            
            logger.info("Starting main loop")
            logger.info("Press SPACE to toggle fullscreen")
            logger.info("Press ESC to quit")
            
            while self.running:
                if not self._handle_events():
                    break
                
                try:
                    if not self.test_mode:
                        # Get current audio position in milliseconds
                        audio_pos = pygame.mixer.music.get_pos()
                        if audio_pos < 0:  # Audio not playing or ended
                            pygame.mixer.music.play(-1)
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_count = 0
                            continue
                        
                        # Calculate target frame based on audio position
                        target_frame = int((audio_pos / 1000.0) * self.fps)
                        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        
                        # Sync frames if needed
                        if abs(target_frame - current_frame) > 2:  # Allow small deviation
                            logger.debug(f"Resyncing: audio frame {target_frame}, video frame {current_frame}")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                            frame_count = target_frame
                        
                        # Read the correct frame
                        ret, frame = self.cap.read()
                        if not ret:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            pygame.mixer.music.play(-1)
                            frame_count = 0
                            continue
                    else:
                        # Test mode handling remains the same
                        clock.tick(60)
                        ret, frame = True, self._create_test_frame()
                    
                    # Process frame
                    if not self.test_mode:
                        frame = cv2.flip(frame, 1)
                        frame_count += 1
                    
                    # Queue frame for processing
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                    except Full:
                        continue
                    
                    # Get processed frame
                    try:
                        processed_frame = self.processed_frame_queue.get(timeout=0.1)
                    except Empty:
                        processed_frame = self._apply_effects(frame)
                    
                    # Display frame
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    display_frame = np.rot90(display_frame)
                    surface = pygame.surfarray.make_surface(display_frame)
                    self.screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    
                    # Update performance metrics
                    self.performance_monitor.update_frame_time()
                    
                    # Maintain frame timing based on audio
                    if not self.test_mode:
                        next_frame_time = (frame_count + 1) * frame_time_ms
                        current_audio_time = pygame.mixer.music.get_pos()
                        sleep_time = (next_frame_time - current_audio_time) / 1000.0
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Starting cleanup...")
        self.running = False
        self.shutdown_event.set()
        
        cleanup_steps = [
            (lambda: pygame.mixer.music.stop() if pygame.mixer.get_init() else None, "audio"),
            (lambda: self.midi_in.close_port() if self.midi_in else None, "MIDI port"),
            (lambda: self.cap.release() if hasattr(self, 'cap') and self.cap.isOpened() else None, "video capture"),
            (lambda: pygame.quit(), "pygame"),
            (lambda: os.remove("audio.wav") if os.path.exists("audio.wav") else None, "temporary audio file")
        ]
        
        for cleanup_func, name in cleanup_steps:
            try:
                cleanup_func()
                logger.info(f"Successfully cleaned up {name}")
            except Exception as e:
                logger.error(f"Error cleaning up {name}: {e}")
        
        logger.info("Cleanup complete")

    def _create_test_frame(self) -> np.ndarray:
        """Create test mode frame"""
        frame = np.full((self.height, self.width, 3), (40, 40, 40), dtype=np.uint8)
        
        if not self.active_effects:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Test Mode - Press MPD218 Pads'
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = self.height // 2
            
            frame = cv2.flip(frame, 0)
            cv2.putText(frame, text, (text_x, text_y), font, 1, (200, 200, 200), 2, cv2.LINE_AA)
            frame = cv2.flip(frame, 0)
        
        return frame


def find_mpd218() -> int:
    """Find the AKAI MPD218 device ID"""
    midi_in = rtmidi.MidiIn()
    ports = midi_in.get_ports()
    
    logger.info("\nAvailable MIDI ports:")
    for i, port in enumerate(ports):
        logger.info(f"{i}: {port}")
        
    for i, port in enumerate(ports):
        if 'MPD218' in port:
            logger.info(f"\nFound MPD218 at port {i}")
            return i
    return 0


if __name__ == "__main__":
    try:
        # Find MPD218
        midi_port = find_mpd218()
        
        # List available displays
        pygame.init()
        displays = pygame.display.get_desktop_sizes()
        print("\nAvailable displays:")
        for i, (width, height) in enumerate(displays):
            print(f"Display {i}: {width}x{height}")
        
        # Get display selection
        display_index = int(input("Enter display number to use (0, 1, etc.): "))
        
        # Ask user for mode
        while True:
            mode = input("Enter mode (1 for video, 2 for test mode): ")
            if mode in ['1', '2']:
                break
            print("Invalid mode. Please enter 1 or 2.")
        
        # Get video path if needed
        video_path = None
        test_mode = (mode == "2")
        if not test_mode:
            video_path = input("Enter path to your video file (e.g., videos/background.mp4): ")
        
        # Create and run VDJ
        vdj = MidiVDJ(
            video_path=video_path,
            midi_port=midi_port,
            test_mode=test_mode,
            display_index=display_index
        )
        
        vdj.run()
        
    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user")
    except Exception as e:
        logger.error(f"Error starting VDJ: {e}")
    finally:
        logger.info("Program terminated")
        sys.exit(0)