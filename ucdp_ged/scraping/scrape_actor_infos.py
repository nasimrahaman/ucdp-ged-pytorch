import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import bs4


ACTOR_URL_TEMPLATE = "https://ucdp.uu.se/actor/{actor_id}"


def fetch_source(url, chrome_driver_path, wait=30):
    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    # chrome_options.add_argument("--no-sandbox") # linux only
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(
        executable_path=chrome_driver_path, options=chrome_options
    )
    driver.get(url)

    time.sleep(wait)
    source = driver.execute_script("return document.body.innerHTML")
    driver.quit()

    return source


def scrape_actor(actor_id, chrome_driver_path):
    url = ACTOR_URL_TEMPLATE.format(actor_id=actor_id)
    source = bs4.BeautifulSoup(fetch_source(url, chrome_driver_path))
    text = [
        str(node).replace("<p>", "").replace("</p>", "")
        for node in source.find_all("p")
        if str(node).startswith("<p>") and str(node).endswith("</p>")
    ]
    death_info = {
        "total": int(source.find_all(id="k-totalDeaths")[0].contents.replace(" ", "")),
        "state": int(
            source.find_all(id="k-stateBasedDeaths")[0].contents.replace(" ", "")
        ),
        "non_state": int(
            source.find_all(id="k-nonStateDeaths")[0].contents.replace(" ", "")
        ),
        "one_sided": int(
            source.find_all(id="k-oneSidedDeaths")[0].contents.replace(" ", "")
        ),
    }
