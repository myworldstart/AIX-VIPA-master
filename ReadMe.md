# AI + X

This is the official repository of VIPA **'AI+X'** Program.

## 运行
    进入项目目录，运行 python run.py  运行项目
    `gunicorn -w 3 -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -b 10.214.211.206:5088 run:app`
