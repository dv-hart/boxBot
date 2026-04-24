"""Perception pipeline — visual detection, re-identification, and voice-visual fusion.

The pipeline is event-driven: motion detection (CPU) triggers person
detection (Hailo), which triggers ReID matching. During conversations,
voice-visual fusion associates speaker diarization with visual detections
using DOA spatial signals. See docs/perception.md.

Module classes are NOT imported here to avoid pulling in heavy
dependencies (cv2, numpy, aiosqlite). Import directly where needed::

    from boxbot.perception.pipeline import PerceptionPipeline, get_pipeline
    from boxbot.perception.clouds import CloudStore
    from boxbot.perception.fusion import IdentityFusion, FusionResult
    from boxbot.perception.voice_reid import VoiceReID
    from boxbot.perception.doa import DOATracker
    from boxbot.perception.crops import CropManager
"""
