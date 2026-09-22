import numpy as np
import pandas as pd
import pytest

from mouse_tracking_utils import extract_distances, load_demo_data, preprocess_data


def test_demo_data_is_packaged_and_extracts_distances():
    data = load_demo_data(2)

    distances = extract_distances(
        data,
        metric="euclidean",
        target_names=["0", "1"],
        squeeze_singleton=False,
    )

    assert len(data) == 8
    assert distances.shape == (2, 2, 2, 2, 20)
    assert np.isfinite(distances).all()


def test_preprocessing_normalises_mouse_tracking_payload():
    data = pd.DataFrame(
        {
            "participant_id": ["participant-1"],
            "mouse_tracking_data": [
                '[{"x": 0, "y": 0, "t": 0}, {"x": 100, "y": 50, "t": 1}]'
            ],
            "screen_width": [100],
            "screen_height": [100],
        }
    )

    preprocessed = preprocess_data(data)

    assert preprocessed.loc[0, "ppt_number"] == "participant-1"
    np.testing.assert_allclose(preprocessed.loc[0, "x_std"], [-0.5, 0.5])
    np.testing.assert_allclose(preprocessed.loc[0, "y_std"], [-0.5, 0.0])


def test_load_demo_data_rejects_unsupported_button_count():
    with pytest.raises(ValueError, match="Unsupported demo dataset size"):
        load_demo_data(3)
