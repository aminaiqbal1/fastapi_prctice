
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles # StaticFiles: This is used to serve downloading files.
import os # os operatig system ka liya use hota hai files ko manege krnyn ka lia
import shutil # sever mn new file banany ka liya shiutil use hota hai
# from pydantic import BaseModel,EmailStr,Field
from fastapi.responses import JSONResponse  # JSONResponse is used to return responses in JSON format
from fastapi.exceptions import RequestValidationError  # Handles validation errors for request data
from fastapi import Request  # Represents an incoming HTTP request

app = FastAPI()  # Creating a FastAPI instance

@app.exception_handler(RequestValidationError)  # Custom exception handler for validation errors
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []  # List to store error details
    for error in exc.errors():  # Iterates over all validation errors
        errors.append({
            "field": ".".join(map(str, error["loc"])),  # Converts error location (tuple) to a string
            "message": error["msg"]  # Retrieves the validation error message
        })
    
    return JSONResponse(  # Returns a JSON response with error details
        status_code=400,  # HTTP 400 Bad Request status code
        content={
            "status": "error",
            "message": "Validation failed",  # General error message
            "errors": errors  # List of detailed validation errors
        }
    )
# upper wala code validation error ko handle krny ka lia hai wo change ni hoga 

upload_folder = "static/uploads"
os.makedirs(upload_folder, exist_ok=True)


app = FastAPI()
#  Mount the uploads folder to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.post("/upload_files/")
async def upload_files(file: UploadFile = File (...)):
    filepath = os.path.join(upload_folder, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
# Return the public URL to access the file
    file_url = f"/static/{file.filename}"
    return {"filename": file.filename, "file_url": file_url}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main1:app", host="0.0.0.0", port=8100, reload=True)