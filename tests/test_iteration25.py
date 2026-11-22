"""Design review question 55: a calibration can reach deployed edges."""

import io
import json

import pytest

from hybrid_edge.gate.confidence import Calibrator

CAL = {"weights": {"mean_prob": 3.0}, "bias": -1.0, "threshold": 0.62, "edge_model": "m"}


class FakeS3:
    def __init__(self, objects):
        self.objects, self.calls = objects, []

    def get_object(self, Bucket, Key):
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)].encode())}


def test_load_from_s3_uri():
    s3 = FakeS3({("models-bucket", "gate/calibration-v3.json"): json.dumps(CAL)})
    cal = Calibrator.load("s3://models-bucket/gate/calibration-v3.json", s3_client=s3)
    assert s3.calls == [("models-bucket", "gate/calibration-v3.json")]
    assert cal.threshold == 0.62 and cal.edge_model == "m" and cal.weights == {"mean_prob": 3.0}


@pytest.mark.parametrize("uri", ["s3://", "s3://bucket", "s3://bucket/", "s3:///key"])
def test_bad_s3_uri_is_rejected(uri):
    with pytest.raises(ValueError, match="bad S3 URI"):
        Calibrator.load(uri, s3_client=FakeS3({}))


def test_local_path_still_works(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(CAL))
    assert Calibrator.load(str(p)).threshold == 0.62


def test_terraform_ships_calibration_with_least_privilege():
    from .test_infra import TF
    ecs = (TF / "modules/edge_region/ecs.tf").read_text()
    assert '{ name = "HEI_GATE_CALIBRATION_PATH", value = var.gate_calibration_s3_uri }' in ecs
    # GetObject on exactly the configured object, not the bucket.
    assert 'Resource = ["arn:aws:s3:::${trimprefix(var.gate_calibration_s3_uri, "s3://")}"]' in ecs
    assert ecs.count("s3:GetObject") == 1
