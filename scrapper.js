function shadowQuery(selector, rootNode = document) {
  const selectors = String(selector).split(">>>");
  let currentNode = rootNode;

  selectors.find((selector, index) => {
    if (index === 0) {
      currentNode = rootNode.querySelector(selectors[index]);
    } else if (currentNode instanceof Element) {
      currentNode = currentNode?.shadowRoot?.querySelector(selectors[index]);
    }

    return currentNode === null;
  });

  if (currentNode === rootNode) {
    return null;
  }

  return currentNode;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitMoreButton() {
  let retries = 0;
  while (
    !shadowQuery('li-library-editor >>> li-button[icon="editor#categories"]')
  ) {
    await sleep(200);
    retries++;
    if (retries > 100) {
      throw Error("Failed to wait More button");
    }
  }
}

async function downloadAllIconsOnPage(opts) {
  const { from, to } = opts || {};

  let icons = Array.from(
    document
      .querySelector("li-library-icons")
      .shadowRoot.querySelectorAll("lord-icon ~ .bg")
  ); //.slice(0, 3);

  if (typeof to === "number" && typeof from === "number") {
    icons = icons.slice(from, to);
  } else if (typeof from === "number") {
    icons = icons.slice(from);
  } else if (typeof to === "number") {
    icons = icons.slice(0, to);
  }

  for (let iconI = 0; iconI < icons.length; iconI++) {
    const icon = icons[iconI];
    // выбираем иконку
    icon.click();
    console.log("icon", iconI, "/", icons.length);
    await sleep(150);

    // получаем список анимаций
    shadowQuery("li-library-editor >>> #state-select >>> #container").click();
    await sleep(200);
    const nStates = shadowQuery(
      "li-overlay-outlet >>> li-field-editor-list"
    ).shadowRoot.querySelectorAll("div").length;
    shadowQuery("li-library-editor >>> #state-select >>> #container").click();
    await sleep(200);

    for (let i = 0; i < nStates; i++) {
      shadowQuery("li-library-editor >>> #state-select >>> #container").click();
      console.log("state", i, "/", nStates);
      await sleep(200);
      shadowQuery("li-overlay-outlet >>> li-field-editor-list")
        .shadowRoot.querySelectorAll("div")
        [i].click();

      // скачиваем анимацию формате mp4
      await waitMoreButton();

      shadowQuery(
        'li-library-editor >>> li-button[icon="editor#categories"]'
      ).click();
      await sleep(100);
      shadowQuery(
        'li-overlay-outlet >>> li-dialog-download >>> li-dialog [data-type="mp4"]'
      ).dispatchEvent(new Event("click", { bubbles: true }));
      await sleep(100);

      let pxInput = shadowQuery(
        'li-overlay-outlet >>> li-dialog-render >>> li-dialog li-field-editor-number[unit="px"] >>> li-input >>> input'
      );
      pxInput.value = 256;
      pxInput.dispatchEvent(new Event("change"));

      let msInput = shadowQuery(
        'li-overlay-outlet >>> li-dialog-render >>> li-dialog li-field-editor-number[unit="ms"] >>> li-input >>> input'
      );
      msInput.value = 0;
      msInput.dispatchEvent(new Event("change"));

      shadowQuery(
        'li-overlay-outlet >>> li-dialog-render >>> li-dialog [icon="editor#download-system"]'
      ).click();

      await sleep(300);

      shadowQuery(
        "li-overlay-outlet >>> li-dialog-render >>> li-dialog >>> span.close"
      ).click();
      shadowQuery(
        "li-overlay-outlet >>> li-dialog-download >>> li-dialog >>> span.close"
      ).click();
      await sleep(300);

      await waitMoreButton();
    }

    // сохраняем svg представление
    shadowQuery(
      'li-library-editor >>> li-button[icon="editor#categories"]'
    ).click();
    await sleep(100);
    shadowQuery(
      'li-overlay-outlet >>> li-dialog-download >>> li-dialog [data-type="svg"]'
    ).dispatchEvent(new Event("click", { bubbles: true }));
    await sleep(100);
    await waitMoreButton();
  }
}

async function main(opts) {
  const { from, to } = opts || {};
  let categories = Array.from(
    shadowQuery("li-library-sidebar >>> .category + div").querySelectorAll(
      "div"
    )
  );

  if (typeof to === "number" && typeof from === "number") {
    categories = categories.slice(from, to);
  } else if (typeof from === "number") {
    categories = categories.slice(from);
  } else if (typeof to === "number") {
    categories = categories.slice(0, to);
  }

  try {
    for (let category of categories) {
      category.click();
      console.log("category", category.innerText);
      await sleep(300);
      // while (!document.querySelector("li-library-icons.busy")) {
      //   await sleep(50);
      // }
      while (document.querySelector("li-library-icons.busy")) {
        await sleep(100);
      }
      await downloadAllIconsOnPage();
    }
  } catch (e) {
    console.error(e);
    alert("Error");
  }
}
