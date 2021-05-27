# Emotion_Detection
I.	Giới thiệu đề tài, những thành phần cần thiết
1.1.	Giới thiệu đề tài
Đề tài này, nhóm sẽ áp dụng máy học, openCV và Python để tạo ra một ứng dụng tự động nhận diện cảm xúc và chọn ra emoji tương ứng từ hai nguồn đầu vào là live video và ảnh chọn từ máy tính.
o	Link Github: https://github.com/HuyDNwebdev/Emotion_Detection
1.2.	Những thành phần cần thiết
1.2.1.	Bộ dữ liệu
-	Bộ dữ liệu sử dụng trong đồ án này là bộ FER-2013 từ Kaggle. Dữ liệu bao gồm hình ảnh thang độ xám 48x48 pixel của khuôn mặt. Các khuôn mặt đã được xử lý để luôn được căn giữa và gần như chiếm cùng một tỉ lệ không gian trong ảnh. Dữ liệu được phân loại từng khuôn mặt dựa trên cảm xúc thể hiện trên nét mặt thành một trong bảy loại (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral) . Tập train bao gồm 28.709 ảnh và tập test bao gồm 3.589 ảnh.
-	Link dữ liệu: https://www.kaggle.com/msambare/fer2013
1.2.2.	Những package cần thiết
-	OpenCV: https://docs.opencv.org/
-	Numpy: https://numpy.org/
-	Keras: https://keras.io/
-	TensorFlow: https://www.tensorflow.org/
-	PyQt5: https://www.riverbankcomputing.com/software/pyqt/
II.	Chuẩn bị data và xây dựng model
Model được train trên Google Colab, file “Train.ipynb”đã được lưu trong Github đề tài.
Các bước train model:
-	Bước 1: Download dữ liệu từ Kaggle:
 
-	Bước 2: Import các thư viện cần thiết:
 
-	Bước 3: Chuẩn bị và xử lý dữ liệu:
 
Xử lý dữ liệu và chuyển dữ liệu từ dạng grayscale sang categorical.
-	Bước 4: Khởi tạo model CNN
 
-	Bước 5: Train và test model:
 
 
Model có accuracy: 0.8610, có thể nói là khá cao.
-	Bước 6: Lưu model vào file “model.h5”
 
III.	Áp dụng model xây dựng GUI nhận dạng biểu tượng cảm xúc từ ảnh và live video.
3.1.	Tạo GUI và chuẩn bị
Tạo một file python đặt cùng thư mục với file “model.h5” và thư mục “emojis”.
 
Bên trong thư mục “emojis” là các hình ảnh biểu tượng cảm xúc với kích thước 400x400 pixcel.
 
Sau đó trong file “final.py” vừa tạo, nhóm bắt đầu lập trình phần giao diện và chức năng cho ứng dụng.
-	Bước 1: Import các thư viện
 
Ngoài các package cơ bản như cv2, numpy hay keras thì package PyQt5 dùng để hỗ trợ tạo giao diện GUI rất dễ sử dụng và tiện lợi, và QfileDialog có tác dụng upload file từ máy tính và get những thông số cần thiết của file như đường dẫn file,…
-	Bước 2: Khởi tạo lại model và load lại trọng số model từ file model.h5
 
-	Bước 3: Tạo 2 dict để lưu nhãn cảm xúc và đường dẫn file ảnh emoji tương ứng
 
-	Bước 4: Từ line 37-189 là code giao diện sử dụng PyQt5, trong đề tài này, nhóm sử dụng các thành của PyQT5 như Qlabel, Qgroupbox, Qpushbutton,… 
 
-	Bước 5: Đây là giao diện ứng dụng, vẫn chưa có code back-end, nên các nút sẽ chưa tương tác được.
 
-	Bước 6: tạo hai hàm fVid và fIMG để thực hiện chức năng nhận diện cảm xúc và chọn emoji tương ứng từ hai nguồn đầu vào là live video và ảnh từ máy tính. Lưu ý, hai hàm này phải nằm trong class Ui_MainWindow để có thể hoạt động với giao diện.
 
Sau khi nhận frame từ webcam, ta sử dụng hàm CascadeClassifier của OpenCV và haarcascade_frontalface_default.xml để nhận diện khuôn mặt, trong khuôn khổ đồ án, chỉ nhận diện khuôn mặt theo hướng chính diện. Ta sẽ sử dụng hàm rectangle của openCV để vẽ một bouding box quanh khuôn mặt.
Sau đó ta xử lý ảnh khuôn mặt được cắt ra từ frame thành dạng phù hợp để có thể dự đoán từ model (chuyển thành dạng grayscale và resize thành 48x48 pixcel).
Đưa ảnh khuôn mặt đã xử lý vào model để dự đoán, mỗi khuôn mặt sẽ có những tỉ lệ cảm xúc khác nhau, ta sẽ dùng np.argmax để chọn cảm xúc có tỉ lệ cao nhất và lưu vào biến maxindex.
Sử dụng hàm putText của OpenCV để đặt text label cảm xúc tương ứng lên bounding box.
Ta sẽ show đồng thời frame với bouding box và label cảm xúc trên live video từ webcam và sử dụng setPixmap và setText của PyQt5 để hiển thị emoji và nhãn cảm xúc tương ứng vào phần hiển thị kết quả của giao diện.
Ta đặt một vòng while gói code lại để có thể nhận diện cảm xúc liên tục, cho đến khi người dùng nhấn “q” từ bàn phím để kết thúc.
Tuy nhiên, hàm ta vừa tạo chưa được liên kết với nút tương ứng trong giao diện. Để có thể liên kết ta thêm dòng code sau vào phần cuối của hàm setupUi trong class Ui_MainWindow.
 
Khi button from Vid được nhấn, sẽ connect tới hàm fVid.
Và đây là kết quả khi ta nhấn vào nút “Từ live video”:
 
Hàm fIMG ta cài đặt gần như tương tự, chỉ khác là sử dụng getOpenFileName từ QFileDialog của PyQt5 để mở ảnh từ máy tính và xử lý ảnh như đã giải thích ở trên.
Và đây là kết quả khi nhấn vào nút “Từ ảnh” và chọn một ảnh từ máy tính:
 
IV.	Hướng dẫn sử dụng và đánh giá ứng dụng
4.1.	Hướng dẫn sử dụng:
Chạy file “final.py” cùng thư mục chứa file “model.h5”, file “haarcascade_frontalface_default.xml” và thư mục “emojis”. Sau đó chọn nguồn ảnh đầu vào là từ live video hay từ ảnh máy tính qua hai nút chức năng và hình ảnh emoji tương ứng sẽ được hiển thị song song ở phần kết quả.
 
4.2.	Đánh giá ứng dụng
-	Ưu điểm: Nhận diện tốt những cảm xúc dễ nhận thấy như tức giận, vui vẻ hay ngạc nhiên, hiển thị kết quả nhanh, độ trễ cực thấp
-	Khuyết điểm: Những cảm xúc như sợ hãi, khinh bỉ,.. thì nhận diện chưa được tốt bởi vì trong thực tế những cảm xúc này mỗi người biểu cảm sẽ có phần khác biệt, nên khi sử dụng trong thực tế, tính ứng dụng chưa cao, cần phải cải tiến thêm. Ngoài ra với những ảnh có góc chụp nghiêng hay có phụ kiện che đáng kể khuôn mặt thì ứng dụng vẫn chưa xử lý tốt được, sẽ cải tiến sau.
