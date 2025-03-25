import qrcode

# 替换为你在 GitHub Pages 上的实际访问链接
website_url = "https://han67890.github.io/yiheng.github.io/"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(website_url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("E:/website_qrcode.png")
print("二维码已生成: website_qrcode.png")
