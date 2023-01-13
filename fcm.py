from pyfcm import FCMNotification
 
APIKEY = "AAAAbywrAbA:APA91bFc1ecgrfEtqBP_vGUuImkiRJ1t3cLwksu2eC8cfNvbQndqkEmz7WECGUrZyKEjL0ybR5lyaW8veRFYLoE56fv7ET8xTOKZGbA12r7Z-uBe_JpKYkgW1teXAVD9wcJLW4AAZ5hU"
 
# 파이어베이스 콘솔에서 얻어 온 서버 키를 넣어 줌
push_service = FCMNotification(APIKEY)
 
def sendMessage(body, title):
    # 메시지 (data 타입)
    data_message = {
        "body": body,
        "title": title
    }
 
    # topic을 이용해 다수의 구독자에게 푸시알림을 전송
    result = push_service.notify_topic_subscribers(topic_name="monitoring", data_message=data_message)
 
    # 전송 결과 출력
    print(result)