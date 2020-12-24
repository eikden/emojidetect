from flask import current_app as app
from ai_api import db
import datetime

#modified title
class Role(db.Model):
  __tablename__ = 'roles'
  id = db.Column(
    db.Integer,
    primary_key=True
  )
  title = db.Column(
    db.String(45),
    nullable=False
  )
  description = db.Column(
    db.Text(),
    nullable=True
  )

  def __init__(self, title):
    self.title = title
