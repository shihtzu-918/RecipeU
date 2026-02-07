# models/database.py
"""
DB 설정 - 기존 db.py 내용
"""
import sqlite3
import json
from typing import Optional, Dict, Any


class RecipeDB:
    """레시피 DB"""
    
    def __init__(self, path: str = "recipes.db"):
        self.path = path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            title TEXT,
            recipe_json TEXT,
            constraints_json TEXT,
            rating INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )          
        """)

        # 기존 DB에 rating 컬럼이 없으면 추가
        cur.execute("PRAGMA table_info(recipes)")
        columns = {row[1] for row in cur.fetchall()}
        if "rating" not in columns:
            cur.execute("ALTER TABLE recipes ADD COLUMN rating INTEGER DEFAULT 0")
        
        conn.commit()
        conn.close()

    def save_recipe(
        self,
        user_id: Optional[str],
        recipe: Dict[str, Any],
        constraints: Dict[str, Any],
        rating: int = 0,
    ) -> int:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO recipes(user_id, title, recipe_json, constraints_json, rating) VALUES (?, ?, ?, ?, ?)",
            (
                user_id,
                recipe.get("title"),
                json.dumps(recipe, ensure_ascii=False),
                json.dumps(constraints, ensure_ascii=False),
                rating,
            ),
        )
        conn.commit()
        recipe_id = cur.lastrowid
        conn.close()
        return recipe_id

    def get_recent(self, user_id: Optional[str], limit: int = 5):
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        if user_id:
            cur.execute(
                "SELECT id, title, created_at, recipe_json, rating FROM recipes WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
        else:
            cur.execute(
                "SELECT id, title, created_at, recipe_json, rating FROM recipes ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        rows = cur.fetchall()
        conn.close()
        return rows

    def get_recipe_by_id(self, recipe_id: int):
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, user_id, title, recipe_json, constraints_json, created_at FROM recipes WHERE id=?",
            (recipe_id,),
        )
        row = cur.fetchone()
        conn.close()
        return row