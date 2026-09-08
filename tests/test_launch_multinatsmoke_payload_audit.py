from fireviewer_model_lab.training.launch_multinatsmoke_payload_audit import build_request


def test_request_is_cpu_bounded_and_non_training():
    request = build_request(run_id="20260906t000000z", script_key="code/audit.py")
    cluster = request["ProcessingResources"]["ClusterConfig"]
    assert cluster == {"InstanceCount": 1, "InstanceType": "ml.t3.large", "VolumeSizeInGB": 30}
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 14400
    assert request["Environment"]["FIREVIEWER_TRAINING_AUTHORIZED"] == "false"
    assert request["AppSpecification"]["ContainerArguments"][-1] == "D-Fire"
