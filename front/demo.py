from flask import Flask, render_template, request, send_file
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        response = requests.get(r'http://202.199.13.24:5001/process', params={'text': text})
        result = response.text.split('\n', 1)[1]
        return render_template('index.html', result=result, text=text)
    return render_template('index.html')

# 设置允许上传的文件类型和最大文件大小
ALLOWED_EXTENSIONS = {'txt'}
MAX_CONTENT_LENGTH = 100 * 1024  # 100 kB

# 检查文件名是否合法
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 上传文件的路由函数
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查上传的文件是否存在
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # 检查文件名是否合法
        if not allowed_file(file.filename):
            return 'Invalid file type'
        # 检查文件大小是否超限
        if request.content_length > MAX_CONTENT_LENGTH:
            return 'File too large'
        # 保存文件到本地
        fpath = './savepath/' + file.filename
        file.save(fpath)
        # 读入字符串
        fexten = file.filename.rsplit('.', 1)[1].lower() # extension
        content = '' # text
        if fexten == 'txt':
            with open(fpath, "r") as f:
                for line in f.readlines():
                    content += line
        # 发给服务器
        response = requests.get(r'http://202.199.13.24:5001/upload_process', params={'text': content})
        processed_fpath = './savepath/processed_' + file.filename
        with open(processed_fpath, "w") as f:
            f.write(response.text)
        return send_file(processed_fpath, as_attachment=True)
    return render_template('index.html')

# 下载文件的路由函数
@app.route('/download', methods=['GET'])
def download_file():
    # 获取文件名
    filename = request.args.get('filename')
    # 检查文件名是否合法
    if not allowed_file(filename):
        return 'Invalid file type'
    # 发送文件到浏览器下载
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
