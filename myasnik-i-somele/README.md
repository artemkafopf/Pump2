# Мясник и Сомелье

Mobile-first веб-приложение для афиши мероприятий, записи клиентов и ручного подтверждения оплат.

## Структура

```text
myasnik-i-somele/
├─ backend/
│  ├─ app/
│  │  ├─ api/
│  │  ├─ core/
│  │  ├─ db/
│  │  ├─ schemas/
│  │  └─ services/
│  ├─ alembic/
│  ├─ Dockerfile
│  └─ requirements.txt
├─ frontend/
│  ├─ src/
│  │  ├─ components/
│  │  ├─ router/
│  │  ├─ services/
│  │  ├─ stores/
│  │  ├─ styles/
│  │  └─ views/
│  ├─ Dockerfile
│  └─ nginx.conf
├─ .env.example
├─ docker-compose.yml
└─ render.yaml
```

## Что реализовано

- FastAPI API с JWT-авторизацией администратора
- PostgreSQL + SQLAlchemy + Alembic
- Публичная афиша, календарь, карточка события, запись и экран оплаты с QR-кодом
- Админ-панель: создание, редактирование, публикация, архивирование, журнал броней и ручное подтверждение оплат
- Защита от переполнения мест через блокировку записи события при бронировании
- Seed-данные и автоматическое создание администратора при старте
- Docker и базовый Render-конфиг

## Локальный запуск

1. Скопируйте `.env.example` в `.env`.
2. При необходимости поменяйте реквизиты оплаты и пароль администратора.
3. Запустите:

```bash
docker compose up --build
```

4. Откройте:

- Frontend: `http://localhost:5173`
- Backend API docs: `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`

## Администратор по умолчанию

- Email: `admin@mis.local`
- Пароль: `Admin123!`

Замените их в `.env` перед публичным деплоем.

## Миграции

```bash
cd backend
alembic upgrade head
python -m app.seed
```

## Деплой на Render

Самый простой вариант для этого проекта:

1. Создать новый Blueprint в Render и выбрать репозиторий с проектом.
2. Использовать `render.yaml` из корня проекта.
3. Задать секреты и платёжные реквизиты через Render Environment.
4. После первого деплоя открыть фронтенд и зайти в админ-панель.

## Production env

- `DATABASE_URL`
- `CORS_ORIGINS`
- `SECRET_KEY`
- `ADMIN_EMAIL`
- `ADMIN_PASSWORD`
- `ADMIN_NAME`
- `PAYMENT_RECIPIENT_NAME`
- `PAYMENT_PHONE`
- `PAYMENT_BANK_NAME`
- `PAYMENT_ACCOUNT_NUMBER`
- `PAYMENT_BIC`
- `PAYMENT_NOTE_PREFIX`
- `FRONTEND_URL`
- `PUBLIC_BASE_URL`
- `SEED_SAMPLE_DATA`
- `VITE_API_BASE_URL`

## Что останется сделать вручную

- Подставить реальные платёжные реквизиты
- Поменять стандартный пароль администратора
- Выбрать и зарегистрировать домен
- Подключить домен к Render или Railway
- При желании заменить текстовый QR-пэйлоад на банковский или SBP-специфичный формат
