# !bin/bash
gunicorn -w 3 -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -b 10.214.29.110:5088 -t 1000  run:app

# --reload                           设置是否修改源代码后, 自动重启

# --error-logfile error_logs.txt     设置项目的报错信息写入一个指定的log文件中

# --log-file logs.txt                设置项目的运行信息写入一个指定的log文件中


