from models.tabels import Post, User, product   
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from config.database import get_db, engine, SessionLocal,Base
Base.metadata.create_all(bind=engine)


app = FastAPI()

class PostCreate(BaseModel):
    
    tittle: str = None
    name: str = None
    user_id: int = None
class PostResponse(PostCreate):
    id: int
    class Config:
        orm_mode = True

class user_create(BaseModel):
    id: int
    gmail: str = None
    password: str = None

    class Config:
        orm_mode = True
@app.post("/users/")
def create_user(user: user_create, db: Session = Depends(get_db)):
    try:
        db_user = User(gmail=user.gmail, password=user.password)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return {"data" :db_user,
                "message": "User created successfully",
                "status": True}
    except Exception as e:
        return {"message": "User not created",
                "status": False,
                "error": str(e)}
@app.post("/posts/{user_id}")
def create_post(user_id,chack: PostCreate, db: Session = Depends(get_db)):
    try:
        db_post = Post(tittle=chack.tittle, name=chack.name, user_id=chack.user_id)
        db.add(db_post)
        db.commit()
        db.refresh(db_post)
        return {"data" :db_post,
                "message": "Post created successfully",
                "status": True}
    except Exception as e:
        return {"message": "Post not created",
                "status": False,
                "error": str(e)}


# Read All Users
@app.get("on/posts/")
def get_posts( db= Depends(get_db)):
    return db.query(Post).all()
# Read User by ID
@app.get("/users/{post_id}")
def get_user(post_id: int, db= Depends(get_db)):
    return db.query(Post).filter(Post.id == post_id).first()

from models.tabels import product
# @app.post("startup")
# def startup_event():
#     db = SessionLocal()
#     # Check if already data exists
#     if db.query(product).count() == 0:
#         products = [
#             product(name="Laptop", description="A high-performance laptop with 16GB RAM", price=1200.0),
#             product(name="Smartphone", description="Latest model with 5G support", price=799.0),
#             product(name="Headphones", description="Noise-cancelling wireless headphones", price=150.0),
#             product(name="Keyboard", description="Mechanical keyboard with RGB lighting", price=90.0)
#         ]
#         db.add_all(products)
#         db.commit()
#     db.close() 

# from fastapi import FastAPI
# from models import SessionLocal, Product

app = FastAPI()

@app.post("startup")
def add_data():
    db = SessionLocal()
    # Agar products table mein koi data nahi hai to insert karo
    if db.query(product).count() == 0:
        products = [
            product(name="Laptop", description="A high-performance laptop with 16GB RAM", price=1200.0),
            product(name="Smartphone", description="Latest model with 5G support", price=799.0),
            product(name="Headphones", description="Noise-cancelling wireless headphones", price=150.0),
            product(name="Keyboard", description="Mechanical keyboard with RGB lighting", price=90.0)
        ]
        db.add_all(products)
        db.commit()
    db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
# this command is use for createing and changeing for every table, column, 

# alembic revision --autogenerate -m "create new column user_id in posts table"
# alembic upgrade head
# alembic revision --autogenerate -m "make relationship of Users with  posts table"