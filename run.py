from app import app
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from asm.utils import seek_gpu

server = pywsgi.WSGIServer(('127.0.0.1', 5088), app, handler_class=WebSocketHandler)
app.run(
    host='127.0.0.1',
    port=5088,
    debug=True,
    threaded=True,
)

server.serve_forever()

if __name__ == '__main__':
	seek_gpu()
	print('启动成功')
	app.run()



