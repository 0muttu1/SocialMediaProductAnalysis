import asyncio
from pyppeteer import launch
import re

# List to store lines for checking repetition
lines_seen = set()

# List of standard words on Google Shopping page to avoid
avoid_words = {"Skip to main content", "Quick SettingsSign in", "Quick Settings", "Sign in", "Images", "Maps", "Shopping", 
                ",Quick SettingsSee all Search settings","Your ActivitySearch CustomisationOffLearn more about your data in Search"

               "Search CustomisationOff", "Learn more about your data in Search", "SafeSearchHelps manage explicit content",
               "ShoppingShopping settingsInformation for MerchantsReport a ViolationShopping help", "Shopping settings",
               "Information for Merchants", "Report a Violation", "Shopping help", "SupportSearch helpSend feedbackNewPrivacyTerms",
               "Search help", "Send feedbackNew", "Privacy", "Terms", "Show only", 
           
 "Delivery and returns",
               "Product rating", "Fewer", "More", "ConditionNew itemsUsed items", "Condition", "New itemsUsed items", "Seller", "About these results",
               "PRICE DROP", "Google apps"}

async def get_text(page, element, file):
    # Get the text content of the current element
    text = await page.evaluate('(element) => element.textContent', element)
    
    # Check if the text contains only alphabets, hyphens, and spaces (case insensitive)
    if re.match('^[a-zA-Z\- ]*$', text):
        # Remove leading and trailing spaces
        text = text.strip()
        # Avoid general Google element header text on shopping page
        if text not in avoid_words:
            # Check for empty lines and repetition of the same line
            if text and text not in lines_seen:
                # Write the text content to the file
                file.write(text + '\n')
                # Add line to the set of seen lines
                lines_seen.add(text)

    # Get all child <div> elements
    children = await element.querySelectorAll(':scope > div')

    # Recursively get the text content of each child element
    for child in children:
        await get_text(page, child, file)

async def scrape(query):
    browser = await launch({
        "executablePath": "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe",
        "headless": True,  # Make it headless
        "args": ["--window-size=800,3200"]
    })

    page = await browser.newPage()
    await page.setViewport({"width": 800, "height": 3200})

    # Perform a Google Shopping search
    await page.goto(f"https://www.google.com/search?tbm=shop&q={query}")

    # Wait for the elements with class ".lg3aE" to load
    await page.waitForSelector(".lg3aE")

    # Select the root element
    root = await page.querySelector('body')

    # Open the text file with name 'queryKeywords.txt'
    with open('Keyword/output.txt', 'w',encoding="utf8") as file:
        # Start the recursion from the root element
        await get_text(page, root, file)

    await browser.close()

# Replace 'YOUR_QUERY' with your actual search query
asyncio.get_event_loop().run_until_complete(scrape('sony xperia'))
print("done")
