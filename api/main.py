import argparse
import uvicorn



if __name__ == '__main__':
    uvicorn.run("api:app", port=8888, reload=True)   
