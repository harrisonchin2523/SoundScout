FROM node:latest as react-build
WORKDIR /app


FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
ADD build /usr/share/nginx/html