# Nộp bài AI Race vòng 2 bằng lệnh `airace`

Vòng 2 không nộp qua web. Mọi thao tác đều bằng lệnh `airace` **chạy trên VM
được cấp cho đội bạn**.

Máy đã được ban tổ chức cấu hình sẵn — không cần đăng nhập, không cần nhập token.

---

## 1. Việc đầu tiên: kiểm tra kết nối

```bash
airace status
```

```
Kết nối OK
  Thí sinh   : Đội 07 - edg3runn3r
  Đội        : 019f13bc-8a15-730a-8731-a34df40e28a6
  Giờ hệ thống: 2026-08-16T09:12:44Z
```

Phải in **đúng tên đội của bạn**. Nếu không, hoặc báo lỗi kết nối, **báo ban tổ
chức ngay** — đừng tự sửa gì trong `/etc/airace/`.

`Giờ hệ thống` là đồng hồ của hệ thống chấm. Mốc hết hạn nộp bài tính theo đồng
hồ này, không phải đồng hồ máy bạn.

---

## 2. Bài 1 (VAR) — nộp file

```bash
airace submit --task var result.zip
```

```
File   : result.zip (1.8 GiB)
sha256 : 3f1c9a2b7d0e4f81...
Mở phiên nộp: 231 mảnh × 8.0 MiB
Mốc nộp bài : 2026-08-16T09:15:02Z  (tính theo lúc bắt đầu, không phải lúc tải xong)
[==================================================] 231/231
xong

ĐÃ NỘP THÀNH CÔNG
  Mã bài nộp : sub-2fa7191b
  Mốc nộp bài: 2026-08-16T09:15:02Z
```

**Mốc nộp bài tính từ lúc bắt đầu tải lên**, không phải lúc tải xong. File lớn
tải mất 20 phút vẫn được ghi nhận theo thời điểm bạn gõ lệnh. Nhưng đừng dựa vào
điều này để bấm lệnh sát giờ — hạn nộp vẫn xét theo mốc bắt đầu, và lệnh gõ sau
hạn thì bị từ chối ngay.

### Mạng đứt giữa chừng

Chạy lại **đúng lệnh cũ**. Hệ thống nối tiếp từ mảnh dang dở, không tải lại từ đầu:

```
Nối tiếp phiên trước: đã có 187/231 mảnh
```

Bấm `Ctrl+C` cũng vậy — không mất gì, chạy lại là tiếp tục.

### Nộp lại đúng file đã nộp

Bị chặn, để tránh tiêu lượt nộp một cách vô ích:

```
Lỗi: file này đã được ghi nhận rồi (mã bài nộp sub-2fa7191b).
  Vẫn muốn nộp lại: thêm --force
```

Chỉ dùng `--force` khi thực sự có lý do — nó **tốn thêm một lượt nộp**.

File khác nội dung (dù trùng tên) thì nộp bình thường, không cần `--force`.

---

## 3. Bài 2 (y tế) và bài 3 (LLM) — đăng ký endpoint

Bạn tự chạy server trên VM của mình, rồi đăng ký địa chỉ:

```bash
airace endpoint --task medical --url http://10.10.1.107:9000
airace endpoint --task llm     --url http://10.10.1.107:8000
```

```
ĐÃ ĐĂNG KÝ ENDPOINT
  Bài         : medical
  Endpoint    : http://10.10.1.107:9000
  Mã bài nộp  : sub-13e28c10
  Thời điểm   : 2026-08-16T09:20:11Z
  Lượt còn lại: 4

Lưu ý: mỗi lần đăng ký là một lần chấm và tính một lượt nộp.
```

### Quy tắc địa chỉ — sai là mất một lượt nộp

| | |
|---|---|
| ✅ `http://10.10.1.107:9000` | đúng |
| ❌ `http://10.10.1.107:9000/` | thừa dấu `/` cuối |
| ❌ `http://10.10.1.107:9000/predict` | hệ thống tự thêm đường dẫn |
| ❌ `10.10.1.107:9000` | thiếu `http://` |
| ❌ `http://127.0.0.1:9000` | hệ thống chấm gọi từ máy khác, không tới được |

**Dùng IP nội bộ của VM bạn, không dùng `localhost`.** Server phải bind
`0.0.0.0`. Tự kiểm tra trước khi đăng ký:

```bash
curl -s http://10.10.1.107:9000/health
```

Không ra kết quả thì đừng gõ lệnh đăng ký — nó vẫn trừ lượt nộp dù endpoint
không gọi được.

### Đăng ký lại

Mỗi lần đăng ký ghi đè endpoint cũ **và tính thêm một lượt nộp**. Server phải
sống trong suốt quá trình chấm; tắt giữa chừng thì lượt đó tính điểm 0.

---

## 4. Xem kết quả

```bash
airace list
```

```
FILE ĐÃ NỘP
  [16/08 09:15] var        result.zip               1.8 GiB  ghi nhận
      mã bài nộp: sub-2fa7191b
      chấm: xong — ĐIỂM 0.8421

ENDPOINT ĐÃ ĐĂNG KÝ
  [16/08 09:20] medical    http://10.10.1.107:9000
      mã bài nộp: sub-13e28c10
      chấm: đang chạy
```

Không cần rời VM để xem điểm — lệnh này hiện luôn trạng thái và điểm.

Các trạng thái chấm:

| Hiển thị | Nghĩa |
|---|---|
| `chưa vào hàng đợi` | Vừa nhận, chưa xếp lịch |
| `đang chờ` | Trong hàng đợi, chờ tới lượt |
| `đang chạy` | Đang chấm |
| `xong — ĐIỂM x.xxxx` | Đã có điểm |
| `lỗi` | Chấm thất bại, kèm lý do ngay dưới |

Bài chấm phải xếp hàng vì số lượt chấm song song có giới hạn. `đang chờ` lâu là
bình thường khi nhiều đội cùng nộp, không phải hỏng.

---

## 5. Lỗi thường gặp

| Thông báo | Xử lý |
|---|---|
| `chưa có thông tin đăng nhập` | Báo BTC. Đừng tự tạo `/etc/airace/credentials` |
| `IP_MISMATCH` | Bạn đang chạy lệnh trên máy khác, không phải VM được cấp |
| `không kết nối được tới hệ thống nộp bài` | Kiểm tra mạng VM; nếu vẫn lỗi, báo BTC |
| `file này đã được ghi nhận rồi` | Nội dung trùng bài đã nộp. Thêm `--force` nếu cố ý |
| `bài nộp bị từ chối — ...` | Hết lượt nộp, hoặc quá hạn. Đọc nội dung đi kèm |
| `file rỗng` | Kiểm tra lại file trước khi nộp |

---

## 6. Tóm tắt lệnh

```
airace status                                Kiểm tra kết nối và tên đội
airace submit --task var <file.zip>          Nộp file bài 1
airace submit --task var <file.zip> --force  Nộp lại file đã ghi nhận
airace endpoint --task medical --url <URL>   Đăng ký endpoint bài 2
airace endpoint --task llm --url <URL>       Đăng ký endpoint bài 3
airace list                                  Xem các lần đã nộp kèm điểm
```

Ba điều đáng nhớ nhất:

- Mạng đứt thì **chạy lại đúng lệnh cũ**, không mất gì.
- Mỗi lần đăng ký endpoint **tốn một lượt nộp**, kể cả khi endpoint sai.
- `airace list` cho biết điểm, không cần hỏi ban tổ chức.
