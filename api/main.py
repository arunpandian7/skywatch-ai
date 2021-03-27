import argparse
import uvicorn
from api import app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", action="store", default="5000")
    args = parser.parse_args()
    port = int(args.port)
    uvicorn.run(app, port=port)
    
