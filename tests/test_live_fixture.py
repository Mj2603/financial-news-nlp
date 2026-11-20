import json
from pathlib import Path

import pandas as pd

from newsnlp.align import assign_session
from newsnlp.live import articles_to_frame

FIXTURE = Path(__file__).parent / "fixtures" / "newsapi_sample.json"


def test_articles_to_frame_from_fixture():
    articles = json.loads(FIXTURE.read_text())["articles"]
    df = articles_to_frame(articles)
    assert len(df) == 2  # untitled row dropped
    assert str(df["published_at"].dt.tz) == "UTC"
    days = pd.DatetimeIndex(["2024-03-04", "2024-03-05"])
    # 14:30Z = 09:30 New York -> Monday; 22:15Z = 17:15 New York -> Tuesday
    assert list(assign_session(df["published_at"], days)) == list(days)
