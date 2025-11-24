import time
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc

def test_driver():
    chrome_driver_path = r"E://Qode_interview//chromedriver-win64//chromedriver-win64//chromedriver.exe"
    user_profile_path = r"C://Users//nehag//AppData//Local//Google//Chrome//User Data//Profile 4"

    print("Initializing Browser...")
    options = uc.ChromeOptions()
    options.add_argument(f"--user-data-dir={user_profile_path}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")

    driver = uc.Chrome(driver_executable_path=chrome_driver_path, options=options)

    try:
        print("Opening Twitter Search...")
        driver.get("https://twitter.com/search?q=%23nifty50&src=typed_query")
        time.sleep(5)  # allow page to load while logged-in

        print("Scrolling...")
        for i in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(4)

        print("Extracting a few tweets...")
        tweets = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')

        print(f"Found tweets: {len(tweets)}")
        for i, t in enumerate(tweets[:5]):
            print(f"{i+1}.", t.text.replace("\n", " ")[:200], "...\n")

    except Exception as e:
        print("❌ ERROR:", e)
    finally:
        print("Closing Browser...")
        driver.quit()


if __name__ == "__main__":
    test_driver()
