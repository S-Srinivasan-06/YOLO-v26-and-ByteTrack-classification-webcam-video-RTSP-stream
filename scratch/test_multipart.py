import email.parser
import email.policy

def parse_multipart(body: bytes, content_type: str):
    msg_bytes = ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode('latin-1') + body
    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(msg_bytes)
    files = {}
    for part in msg.iter_parts():
        fn = part.get_filename()
        if fn:
            files[fn] = part.get_payload(decode=True)
    return files

bnd = '----12345'
ct = 'multipart/form-data; boundary=' + bnd
body = (
    '--' + bnd + '\r\n'
    'Content-Disposition: form-data; name="file"; filename="sample.mp4"\r\n'
    'Content-Type: video/mp4\r\n\r\n'
    'BINARY_VIDEO_CONTENT_PAYLOAD'
    '\r\n--' + bnd + '--\r\n'
).encode('latin-1')

res = parse_multipart(body, ct)
print('Parsed:', list(res.keys()), res.get('sample.mp4'))
assert res.get('sample.mp4') == b'BINARY_VIDEO_CONTENT_PAYLOAD'
print('SUCCESS: Python stdlib handles multipart video upload without any extra packages!')
