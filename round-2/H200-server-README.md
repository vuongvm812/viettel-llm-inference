# @Thông tin truy cập hệ thống

Tài liệu này  hướng dẫn truy cập của cuộc thi:

1. **Luồng Đội thi (tenant)** — mỗi đội (`teamNN`) truy cập đúng 1 VM của mình.

---

## Luồng 1: Đội thi (`teamNN`) truy cập VM của mình

Tài liệu cung cấp cho các đội thi (`teamNN`) và 2 file/1 password cấp riêng cho từng đội. Thay `team01` bên dưới bằng đúng số team của bạn (`team02`, `team03`, ... `team24`).

### Bạn được cấp những gì

| Items | Ở đâu | Dùng để |
|---|---|---|
| 1 file private key (`teamNN`, không có đuôi `.pub`) | BTC gửi riêng, KHÔNG chia sẻ | Đăng nhập SSH |
| 1 mật khẩu (dòng `teamNN <password>` trong danh sách BTC gửi) | BTC gửi riêng | Đăng nhập web IDE (Basic Auth) |
| 2 địa chỉ web | Trong tài liệu này (thay `teamNN` đúng số của bạn) | Code editor + Jupyter |

Địa chỉ web của bạn:
- Code editor: `https://code.teamNN.171.226.125.255.nip.io/`
- JupyterLab: `https://jupyter.teamNN.171.226.125.255.nip.io/`

(`171.226.125.255` là địa chỉ chung, không đổi giữa các team — chỉ `teamNN` là khác nhau.)

### 1. Cài SSH key

Đặt file private key BTC gửi vào `~/.ssh/`, đặt tên đúng theo team, và giới hạn quyền:

```bash
cp teamNN ~/.ssh/teamNN
chmod 600 ~/.ssh/teamNN
```

Trên Windows dùng OpenSSH (PowerShell) hoặc WSL, đường dẫn tương ứng là `%USERPROFILE%\.ssh\teamNN`.

### 2. Cấu hình SSH (khuyên dùng — gõ lệnh ngắn hơn)

Thêm đoạn sau vào `~/.ssh/config` (thay `teamNN` bằng đúng số của bạn ở **cả 6 chỗ**):

```
Host contest-gw-teamNN
    HostName 171.226.125.255
    User teamNN
    IdentityFile ~/.ssh/teamNN
    IdentitiesOnly yes

Host teamNN
    HostName <IP nội bộ VM của bạn, BTC cấp kèm private key>
    User teamNN
    IdentityFile ~/.ssh/teamNN
    IdentitiesOnly yes
    ProxyJump contest-gw-teamNN
    RequestTTY yes
```

Sau đó chỉ cần gõ:

```bash
ssh teamNN
```

Không muốn sửa `~/.ssh/config` thì gõ thẳng (thay `<IP VM>` bằng IP BTC cấp):

```bash
ssh -J teamNN@171.226.125.255 -i ~/.ssh/teamNN teamNN@<IP VM>
```

### 3. Những gì làm được / không làm được qua SSH

**Làm được**: mở terminal tương tác duy nhất (`ssh teamNN`) để chạy lệnh, train
model, quản lý file bằng dòng lệnh trên chính VM của bạn.

**KHÔNG được làm**:
- `scp`, `sftp`, `rsync` để copy file qua SSH — bị từ chối ngay (`Only an
  interactive SSH terminal is allowed...`). **Upload/download file phải qua
  web IDE (mục 4)**, không qua SSH.
- `ssh teamNN "some commnd"` (chạy lệnh không mở terminal) — bị từ chối tương tự.
- `ssh -L`, `-D`, `-R` (port forwarding, SOCKS proxy) — bị từ chối
- SSH sang VM của team khác —  hệ thống chỉ cho mỗi team đi đúng 1 đường tới đúng VM của mình.

Đây là thiết kế của cuộc thi.

### 4. Web IDE — coding, upload file

Dùng trình duyệt, vào 2 địa chỉ ở trên. Trình duyệt sẽ hỏi **user/password
(Basic Auth)** — nhập `teamNN` và mật khẩu BTC cấp riêng cho bạn. Không cần
bấm qua cảnh báo "Not secure" gì cả — trang dùng chứng chỉ HTTPS thật (Let's
Encrypt), trình duyệt tự tin luôn. **Nếu bạn thấy cảnh báo "Not secure" / "Your
connection is not private"**, đó là bất thường — báo BTC ngay, đừng bấm qua.

Sau khi vào được:
- **Code editor** (`code.teamNN...`): VS Code chạy trên trình duyệt, làm việc
  trực tiếp trên `/srv/contest-workspace` của VM bạn. Kéo-thả file vào cửa sổ
  editor để upload.
- **JupyterLab** (`jupyter.teamNN...`): mở/chạy notebook như bình thường.
##  5. Proxy

VM không có Internet trực tiếp. Cài package phải qua proxy nội bộ `10.10.1.126:3128` — **đã cấu hình sẵn**, không cần làm gì thêm với `apt`, `pip`, `npm`, `conda`, `docker pull`.

Nếu 1 tool không tự đọc được proxy có thể set:

```bash 
curl -x http://10.10.1.126:3128 https://pypi.org/simple/
export HTTP_PROXY=http://10.10.1.126:3128 HTTPS_PROXY=http://10.10.1.126:3128
```
### 6. Sự cố thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Permission denied (publickey)` khi SSH | Sai key, hoặc quên `chmod 600` | Kiểm tra đúng file `teamNN`, đã `chmod 600` |
| SSH bị treo / timeout ở bước kết nối | Có thể là lỗi Gateway | Đợi vài phút rồi thử lại; báo BTC nếu kéo dài |
| Web báo `401 Unauthorized` | Sai user/password Basic Auth | Kiểm tra lại đúng `teamNN` + password BTC cấp, phân biệt hoa/thường |
| `scp`/`sftp` báo lỗi / bị treo | Chặn theo thiết kế | Dùng web IDE để upload |
| Không ra được Internet từ VM | VM không có egress ngoại trừ một số endpoint do BTC cấu hình | Báo BTC nếu có yêu cầu cài gói |

### 7. Lưu ý

- Không chia sẻ private key hay password của bạn cho team khác — hệ thống
  ghi log chi tiết mọi kết nối SSH (kể cả các lần bị chặn).
- Không có Internet trên VM — chỉ có đường tới proxy/gói cài đặt và API nộp
  bài do BTC cấu hình sẵn.

---
