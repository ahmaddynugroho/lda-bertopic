# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import scrapy
import re


def clean(str):
    match = ["<.*?>", "\r", "\n", "\t", "\s\s+"]
    match = [re.compile(m) for m in match]
    for m in match:
        str = re.sub(m, " ", str)
    return str


class Tempo(scrapy.Spider):
    name = "tempo"
    start_urls = [
        f"https://www.tempo.co/indeks/2022-12-{1 + day:02}" for day in range(31)
    ]

    def parse(self, response):
        links = response.css(".title a::attr(href)").getall()
        for a in links:
            yield scrapy.Request(a, callback=self.parse_article)

    def parse_article(self, response):
        headline = response.css(".title::text").get().strip()
        body = response.css(".detail-konten>p *::text").getall()
        body = [p.replace("\xa0", " ").strip() for p in body]
        body = " ".join(body)
        if headline and body:
            yield {"headline": headline, "body": body}


class Digilib(scrapy.Spider):
    name = "digilib"
    # start_urls = [
    #     "https://digilib.uin-suka.ac.id/view/divisions/jur=5Ftinf/2010.html"
    # ]
    start_urls = [
        f"https://digilib.uin-suka.ac.id/view/divisions/jur=5Ftinf/20{1 + yy}.html"
        for yy in range(9, 21)
    ]

    def parse(self, r):
        links = r.css(".ep_view_page p a::attr(href)").getall()
        for a in links:
            yield scrapy.Request(a, callback=self.parse_abstract)

    def parse_abstract(self, r):
        title = r.css("h1::text").get().strip()
        abstract = r.css(".ep_summary_content_main h2 + p::text").get().strip()
        text = f"{title}. {abstract}"
        text = clean(text)
        yield {"digilib": text}


class Ijid(scrapy.Spider):
    name = "ijid"
    start_urls = [  # got this by scrapy shell
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/285",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/271",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/261",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/242",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/234",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/225",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/206",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/168",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/166",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/160",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/159",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/158",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/185",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/184",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/183",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/182",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/181",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/180",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/179",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/178",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/18",
        "https://ejournal.uin-suka.ac.id/saintek/ijid/issue/view/9",
    ]

    def parse(self, r):
        articles = r.css(
            'h2:contains("Articles") ~ .article-summary .article-summary-title a::attr(href)'
        ).getall()
        for article_link in articles:
            yield scrapy.Request(article_link, callback=self.parse_article)

    def parse_article(self, r):
        title = r.css("h1::text").get().strip()
        abstract = r.css(".article-details-abstract").get().strip()
        text = f"{title}. {abstract}"
        text = clean(text).replace("Abstract", "").replace("  ", "")
        yield {"ijid": text}
