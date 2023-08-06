# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import scrapy


class Tempo(scrapy.Spider):
    name = 'tempo'
    start_urls = [
        f'https://www.tempo.co/indeks/2022-12-{1 + day:02}' for day in range(31)
    ]

    def parse(self, response):
        links = response.css('.title a::attr(href)').getall()
        for a in links:
            yield scrapy.Request(a, callback=self.parse_article)

    def parse_article(self, response):
        headline = response.css('.title::text').get().strip()
        body = response.css('.detail-konten>p *::text').getall()
        body = [p.replace('\xa0', ' ').strip() for p in body]
        body = ' '.join(body)
        if headline and body:
            yield {
                'headline': headline,
                'body': body
            }
