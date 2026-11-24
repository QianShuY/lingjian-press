"""Local recipes and inspection evidence, stored transactionally in SQLite."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import cv2
import numpy as np

from inspection import FeedCheck, Hole, read_image


DEMO_ID = "demo"
FEED_DEMO_ID = "feed-demo"
BUILTIN_IDS = (DEMO_ID, FEED_DEMO_ID)
FEED_MARGIN_LEFT = 165
FEED_MARGIN_OTHER = 55
FEED_DEMO_BG = (54, 47, 42)
EVENT_LIMIT = 200
WORK_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    holes: tuple[Hole, ...]
    zones: tuple[tuple[int, int, int, int], ...]
    tolerance: float
    review: tuple[tuple[int, int], ...] = ()
    anchor: tuple[int, int, int, int] | None = None
    feed: FeedCheck | None = None

    @classmethod
    def from_json(cls, payload):
        recipe = json.loads(payload)
        recipe["holes"] = tuple(Hole(**hole) for hole in recipe["holes"])
        recipe["zones"] = tuple(tuple(zone) for zone in recipe["zones"])
        recipe["review"] = tuple(tuple(point) for point in recipe["review"])
        if recipe.get("anchor") is not None:
            recipe["anchor"] = tuple(recipe["anchor"])
        if recipe.get("feed") is not None:
            feed = recipe["feed"]
            recipe["feed"] = FeedCheck(Hole(**feed["hole"]), feed["direction"], feed["tolerance"])
        return cls(**recipe)


def demo_recipe():
    red = (
        (876, 392), (916, 391), (935, 445), (917, 704), (852, 785), (1140, 433),
        (1383, 391), (1427, 391), (1369, 708), (1493, 779), (909, 26), (943, 26),
        (960, 65), (894, 157), (892, 321), (1139, 60), (1140, 292), (1351, 21),
        (1389, 20), (1432, 153), (1337, 259), (1441, 305),
    )
    dark = ((972, 625), (1354, 484))
    holes = tuple(
        Hole(f"孔 {index:02d}", x - 776, y, "red" if index <= len(red) else "dark")
        for index, (x, y) in enumerate(red + dark, 1)
    )
    review = tuple((x - 776, y) for x, y in ((946, 261), (853, 544), (857, 564), (1482, 546), (1479, 570)))
    zones = tuple(
        (x1 - 776, y1, x2 - 776, y2)
        for x1, y1, x2, y2 in ((915, 124, 1008, 212), (1060, 102, 1271, 246),
                               (1321, 75, 1400, 173), (1055, 482, 1303, 677))
    )
    return Recipe(DEMO_ID, "级进模试样", holes, zones, 5.0, review, (396, 408, 554, 554))


def demo_reference():
    source = WORK_ROOT / "assets" / "reference.png"
    if source.is_file():
        return read_image(source)
    recipe = demo_recipe()
    image = np.full((835, 781, 3), (35, 28, 118), dtype=np.uint8)
    cv2.rectangle(image, (54, 3), (751, 811), (214, 219, 223), -1)
    left, top, right, bottom = recipe.anchor
    fixture = np.array([
        (left+13, top+15), (right-16, top+15), (right-16, top+54),
        (right-49, top+54), (right-49, bottom-15), (left+29, bottom-15),
        (left+29, top+87), (left+13, top+87),
    ], dtype=np.int32)
    cv2.fillPoly(image, [fixture], (130, 147, 164))
    for point, radius in (((left+43, top+40), 9), ((right-33, top+34), 7), ((left+62, bottom-40), 11)):
        cv2.circle(image, point, radius, (78, 92, 111), -1)
    for hole in recipe.holes:
        color = (33, 26, 130) if hole.method == "red" else (59, 62, 64)
        cv2.circle(image, (hole.x, hole.y), 7, color, -1)
    return image


def feed_demo_sample(source):
    pad = FEED_MARGIN_OTHER
    frame = cv2.copyMakeBorder(source, pad, pad, FEED_MARGIN_LEFT, pad, cv2.BORDER_CONSTANT, value=FEED_DEMO_BG)
    # The fixture mark is drawn in a separate area so it never moves with the sample strip.
    points = np.array([(34, 324), (118, 324), (118, 382), (91, 382), (91, 453), (34, 453)], dtype=np.int32)
    cv2.fillPoly(frame, [points], (181, 193, 201))
    cv2.polylines(frame, [points], True, (107, 121, 129), 2, cv2.LINE_AA)
    for center, size in (((54, 346), 8), ((97, 351), 11), ((58, 424), 10)):
        cv2.circle(frame, center, size, (43, 49, 55), -1, cv2.LINE_AA)
        cv2.line(frame, (center[0]-5, center[1]), (center[0]+5, center[1]), (124, 137, 145), 2, cv2.LINE_AA)
    cv2.line(frame, (52, 386), (76, 373), (60, 69, 73), 4, cv2.LINE_AA)
    hole = Hole("送料定位孔", FEED_MARGIN_LEFT + 364, pad + 433, "red")
    recipe = Recipe(FEED_DEMO_ID, "送料到位演示", (), (), 5.0, anchor=(24, 310, 133, 466),
                    feed=FeedCheck(hole, "down", 5.0))
    return recipe, frame


def encode(image, ext):
    if image is None:
        return None
    success, data = cv2.imencode(ext, image)
    if not success:
        raise ValueError("图片无法保存")
    return data.tobytes()


def decode(value):
    if value is None:
        return None
    return cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)


class Repository:
    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "inspection.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS recipes (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, config TEXT NOT NULL, reference BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created TEXT NOT NULL, run_id TEXT NOT NULL, cycle INTEGER NOT NULL,
                    recipe TEXT NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL,
                    elapsed_ms REAL NOT NULL, original BLOB, marked BLOB
                );
            """)
            demo = db.execute("SELECT config FROM recipes WHERE id=?", (DEMO_ID,)).fetchone()
            if demo is None:
                self.store_recipe(db, demo_recipe(), encode(demo_reference(), ".png"))
            else:
                recipe = Recipe.from_json(demo["config"])
                if recipe.anchor is None:
                    updated = replace(recipe, anchor=demo_recipe().anchor)
                    db.execute("UPDATE recipes SET config=? WHERE id=?",
                               (json.dumps(asdict(updated), ensure_ascii=False), DEMO_ID))
            if db.execute("SELECT 1 FROM recipes WHERE id=?", (FEED_DEMO_ID,)).fetchone() is None:
                recipe, frame = feed_demo_sample(demo_reference())
                self.store_recipe(db, recipe, encode(frame, ".png"))
            db.execute("INSERT OR IGNORE INTO settings VALUES ('active_recipe', ?)", (DEMO_ID,))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def store_recipe(db, recipe, encoded):
        db.execute("INSERT INTO recipes VALUES (?, ?, ?, ?)",
                   (recipe.id, recipe.name, json.dumps(asdict(recipe), ensure_ascii=False), encoded))

    def save_recipe(self, recipe, reference):
        encoded = encode(reference, ".png")
        with self.connection() as db:
            db.execute(
                """INSERT INTO recipes VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, config=excluded.config, reference=excluded.reference""",
                (recipe.id, recipe.name, json.dumps(asdict(recipe), ensure_ascii=False), encoded),
            )
            db.execute("UPDATE settings SET value=? WHERE name='active_recipe'", (recipe.id,))

    def recipes(self):
        with self.connection() as db:
            return db.execute("SELECT id, name FROM recipes ORDER BY rowid DESC").fetchall()

    def activate(self, recipe_id):
        with self.connection() as db:
            db.execute("UPDATE settings SET value=? WHERE name='active_recipe'", (recipe_id,))

    def load_active(self):
        with self.connection() as db:
            row = db.execute("SELECT value FROM settings WHERE name='active_recipe'").fetchone()
        return self.get_recipe(row["value"])

    def load_replay_setup(self):
        with self.connection() as db:
            row = db.execute("SELECT value FROM settings WHERE name='replay_setup'").fetchone()
            return row["value"] if row is not None else None

    def save_replay_setup(self, payload):
        with self.connection() as db:
            db.execute(
                "INSERT INTO settings VALUES ('replay_setup', ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                (payload,),
            )

    def get_recipe(self, recipe_id):
        with self.connection() as db:
            row = db.execute("SELECT config, reference FROM recipes WHERE id=?", (recipe_id,)).fetchone()
        if row is None:
            raise ValueError("未找到该模具配方")
        image = decode(row["reference"])
        if image is None:
            raise ValueError("良品图无法打开")
        return Recipe.from_json(row["config"]), image

    def add_event(self, run_id, cycle, recipe_name, verdict, elapsed, original):
        if verdict.status == "ok":
            return
        source_jpg = encode(original, ".jpg")
        marked_jpg = encode(verdict.image, ".jpg")
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        with self.connection() as db:
            db.execute(
                """INSERT INTO events (created, run_id, cycle, recipe, status, reason, elapsed_ms, original, marked)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, run_id, cycle, recipe_name, verdict.status, verdict.reason, elapsed, source_jpg, marked_jpg),
            )
            db.execute("DELETE FROM events WHERE id <= (SELECT MAX(id) - ? FROM events)", (EVENT_LIMIT,))

    def events(self):
        with self.connection() as db:
            return db.execute(
                "SELECT id, created, cycle, recipe, status, reason FROM events ORDER BY id DESC"
            ).fetchall()

    def event(self, event_id):
        with self.connection() as db:
            return db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
