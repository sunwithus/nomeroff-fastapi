Соберите Docker-образ
# Авторизуйтесь на Docker Hub (если ещё не авторизованы)
docker login

# Соберите образ (замените username на ваш логин на Docker Hub)
docker build -t username/nomeroff-net:latest .

# Можно добавить версию
docker build -t username/nomeroff-net:v1.0.0 .


Проверьте локально
# Запустите контейнер для проверки
docker run -p 8080:8080 username/nomeroff-net:latest

# Откройте другой терминал и проверьте работу
curl http://localhost:8080
# или откройте в браузере http://localhost:8080


Загрузите на Docker Hub
# Загрузите образ с тегом latest
docker push username/nomeroff-net:latest

# Загрузите образ с версией
docker push username/nomeroff-net:v1.0.0


Проверка загрузки
# Очистите локальные образы (опционально)
docker rmi username/nomeroff-net:latest

# Скачайте и запустите образ с Docker Hub
docker pull username/nomeroff-net:latest
docker run -p 8080:8080 username/nomeroff-net:latest