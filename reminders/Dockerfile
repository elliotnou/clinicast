FROM ruby:3.3-bookworm

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
      build-essential libpq-dev postgresql-client curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Gemfile Gemfile.lock ./
RUN bundle install

COPY . .

EXPOSE 3000

CMD ["bundle", "exec", "puma", "-C", "config/puma.rb"]
