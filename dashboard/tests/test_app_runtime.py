import sys
import warnings

import pytest


pytestmark = pytest.mark.filterwarnings("ignore:Type google.protobuf.pyext._message.*:DeprecationWarning")


def test_default_dashboard_screen_renders_with_real_csv():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf.*")
    from streamlit.testing.v1 import AppTest

    old_argv = sys.argv[:]
    sys.argv = ["dashboard/app.py", "--csv", "us/breakout_follow_pool.csv"]
    try:
        app = AppTest.from_file("dashboard/app.py", default_timeout=10).run(timeout=30)
    finally:
        sys.argv = old_argv

    assert len(app.exception) == 0
    assert [title.value for title in app.title] == []
    assert any("Current Filters" in item.value for item in app.subheader)
    assert len(app.selectbox) > 0
