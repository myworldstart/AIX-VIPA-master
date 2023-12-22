from app import app
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import sys
from asm.utils import seek_gpu

sys.path.append("yolo")

server = pywsgi.WSGIServer(('10.214.211.207', 5088), app, handler_class=WebSocketHandler)
app.run(
    host='10.214.211.207',
    port=5088,
    debug=True,
    threaded=True,
)

server.serve_forever()

if __name__ == '__main__':
	seek_gpu()
	print('启动成功')
	app.run()



