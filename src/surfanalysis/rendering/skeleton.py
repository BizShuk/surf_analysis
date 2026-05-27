"""MediaPipe Pose skeleton connectivity (which keypoints connect to which)."""

from __future__ import annotations

from surfanalysis.extraction.landmarks import (
    L_ANKLE, L_EAR, L_ELBOW, L_EYE, L_EYE_INNER, L_EYE_OUTER,
    L_FOOT, L_HEEL, L_HIP, L_INDEX, L_KNEE, L_MOUTH, L_PINKY,
    L_SHOULDER, L_THUMB, L_WRIST, NOSE,
    R_ANKLE, R_EAR, R_ELBOW, R_EYE, R_EYE_INNER, R_EYE_OUTER,
    R_FOOT, R_HEEL, R_HIP, R_INDEX, R_KNEE, R_MOUTH, R_PINKY,
    R_SHOULDER, R_THUMB, R_WRIST,
    TOTAL_LANDMARKS,
)

SKELETON_EDGES: list[tuple[int, int]] = [
    # face
    (NOSE, L_EYE_INNER), (L_EYE_INNER, L_EYE), (L_EYE, L_EYE_OUTER), (L_EYE_OUTER, L_EAR),
    (NOSE, R_EYE_INNER), (R_EYE_INNER, R_EYE), (R_EYE, R_EYE_OUTER), (R_EYE_OUTER, R_EAR),
    (L_MOUTH, R_MOUTH),
    # arms
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (L_WRIST, L_PINKY), (L_WRIST, L_INDEX), (L_WRIST, L_THUMB), (L_PINKY, L_INDEX),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (R_WRIST, R_PINKY), (R_WRIST, R_INDEX), (R_WRIST, R_THUMB), (R_PINKY, R_INDEX),
    # torso
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    # legs
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT), (L_ANKLE, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT), (R_ANKLE, R_FOOT),
]


def valid_indices() -> list[int]:
    return list(range(TOTAL_LANDMARKS))
