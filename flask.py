# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 19:24:33 2021

@author: Vishal pc
"""
from flask import Flask

app = Flask(__name__)

@app.route("/")

def hello():
    return "welcome to canada."

if __name__ == '__main__':
    app.run(debug=True, port=12345)