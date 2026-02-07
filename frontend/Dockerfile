# Build stage
FROM node:18-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# 빌드된 파일을 nginx로 복사 (dist로 수정!)
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx 설정 파일 (필요시)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
