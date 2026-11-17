"""Check the real plugin/selector with GPU hidden and SM metadata stubbed.

No backend instance, CUDA kernel or model is executed. This is a registration
check only, not GPU inference validation.
"""

import json
import os
import tempfile


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run this import-only check with CUDA_VISIBLE_DEVICES=''")
    # FastDeploy creates dated log files during import. Keep this read-only
    # contract check from polluting the submission source tree.
    temporary_logs = tempfile.TemporaryDirectory(prefix="bts-fastdeploy-registration-")
    os.environ["FD_LOG_DIR"] = temporary_logs.name
    from bts_fastdeploy.environment import prepare_workers
    prepare_workers()
    import fastdeploy.model_executor.utils as utils
    # The release initializes its FA implementation at import time. Supply the
    # intended SM90 metadata without creating a CUDA context on a busy device.
    utils.get_sm_version = lambda: 90
    os.environ["FD_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["BTS_FASTDEPLOY_ENABLE"] = "1"
    from bts_fastdeploy.plugin import register
    register()
    assert os.environ["FD_DISABLE_CHUNKED_PREFILL"] == "1"
    import paddle
    from fastdeploy.model_executor.layers.attention import flash_attn_backend

    expected_attention_contract = {
        "fastdeploy_flash_attn_version": 2,
        "paddle_flash_attn_version": 2,
        "dense_prefill_op": "flash_attn_unpadded",
        "decode_op": "append_attention",
    }
    assert flash_attn_backend.FLASH_ATTN_VERSION == 2
    assert (
        paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
            "FLAGS_flash_attn_version"
        ]
        == 2
    )
    assert (
        flash_attn_backend.FlashAttentionBackend._bts_native_attention_contract
        == expected_attention_contract
    )
    from fastdeploy.model_executor.layers.attention.attention_selecter import get_attention_backend
    cls = get_attention_backend()
    assert cls._bts_fastdeploy_adapter
    register()
    assert get_attention_backend() is cls
    print(json.dumps(dict(status="passed", selected_backend=cls.__name__,
                         gpu_execution=False, sm_metadata_stubbed=90,
                         native_attention_contract=expected_attention_contract,
                         chunked_prefill_guard="FD_DISABLE_CHUNKED_PREFILL=1"),indent=2))


if __name__ == "__main__":
    main()
