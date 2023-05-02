import React, { KeyboardEventHandler, MouseEventHandler } from "react";
import "./App.css";
import "./iframe-api";
declare global {
  interface Window {
    onSpotifyIframeApiReady: any;
  }
}

let ready: boolean = false;

let IFrameAPI: any = null;
let EmbedController: any = null;

let result: String[][] = [];
let rel_track_list: String[][] = [];
let irrel_track_list: String[][] = [];

let selected: number = 0;
let rated: number[] = [];

function App() {
  return (
    <>
      <div className="parallax-wrap">
        <div className="background-grid"></div>
      </div>
      <main>
        <div className="top-text">
          <div id="title">
            <h1>
              <span style={{ color: "deepskyblue" }}>Sound</span>Scout
            </h1>
          </div>
          <div className="input-box" onClick={sendFocus}>
            <img
              src="mag.png"
              onClick={search}
              // onKeyDown={handlePress}
              alt="Search"
            />
            <input placeholder="Search for a playlist" id="filter-text-val" />
          </div>
        </div>
        <div id="result">
          <div id="left" />
          <div id="right">
            <div id="iframe">
              <div id="embed-iframe" />
            </div>
            <p id="thanks">Thank you for the feedback!</p>
            <div id="roc">
              <div onClick={rocUp}>👍</div>
              <div onClick={rocDown}>👎</div>
            </div>
            <div id="regen" onClick={regen}>
              Regenerate Results
            </div>
          </div>
        </div>
      </main>
      <footer>
        Created by Harrison Chin, Willy Jiang, Joshua Guo, Eric Huang, Alex
        Levinson
      </footer>
    </>
  );
}

function songTemplate(song: string[]) {
  return `<p>${song[0]}</p><p>${song[1]}</p>`;
}

const sendFocus: MouseEventHandler<HTMLDivElement> = (e) => {
  (document.getElementById("filter-text-val") as HTMLInputElement).focus();
};

function clear() {
  const main: HTMLElement = document.getElementById("result") as HTMLDivElement;
  main.style.display = "grid";
  const box: HTMLElement = document.getElementById("iframe") as HTMLDivElement;
  while (box.lastElementChild) {
    box.removeChild(box.lastElementChild);
  }
  let temp = document.createElement("div");
  temp.id = "embed-iframe";
  box.appendChild(temp);
  const boxbox: HTMLElement = document.getElementById("left") as HTMLDivElement;
  while (boxbox.lastElementChild) {
    boxbox.removeChild(boxbox.lastElementChild);
  }
  result = [];
  rel_track_list = [];
  irrel_track_list = [];
  EmbedController = null;
}

const search: MouseEventHandler<HTMLImageElement> = (e) => {
  checkReady();
  clear();
  (document.getElementById("title") as HTMLDivElement).style.display = "none";
  fetch(
    "http://4300showcase.infosci.cornell.edu:4522/search?" +
      // "http://localhost:5000/search?" +
      new URLSearchParams({
        title: (document.getElementById("filter-text-val") as HTMLInputElement)
          .value,
      }).toString()
  )
    .then((response) => response.json())
    .then((data) =>
      data.forEach((row: string[], i: number) => {
        // each row is [song name, song artist, song uri]
        result[i] = row;
        let tempDiv = document.createElement("div");
        tempDiv.setAttribute("data-id", i.toString());
        tempDiv.id = i.toString();
        tempDiv.onclick = function(e) {
          const element = e.target as HTMLElement;
          if (
            (document.getElementById(selected.toString()) as HTMLDivElement)
              .classList != null
          ) {
            (document.getElementById(
              selected.toString()
            ) as HTMLDivElement).classList.remove("selected");
          }
          element.classList.add("selected");
          const i = parseInt(element.getAttribute("data-id") || "0");
          selected = i;
          EmbedController.loadUri(result[i][2]);
        };
        tempDiv.innerHTML = songTemplate(row);
        const doc = document.getElementById("left") as HTMLElement;
        doc.appendChild(tempDiv);
      })
    )
    .then(() => {
      if (IFrameAPI == null) {
        // MAKE IT WAIT
      }
      // console.log(result);
      const element = document.getElementById("embed-iframe");
      const options = {
        width: "400",
        height: "400",
        uri: result[0][2],
      };
      const callback = (controller: any) => {
        EmbedController = controller;
      };
      IFrameAPI.createController(element, options, callback);
    });
};

const parallax = (event: MouseEvent) => {
  const x = (window.innerWidth - event.pageX * 1) / 90;
  const y = (window.innerHeight - event.pageY * 1) / 90;

  (document.querySelector(
    ".parallax-wrap .background-grid"
  ) as HTMLDivElement).style.transform = `translateX(${x}px) translateY(${y}px)`;
};

document.addEventListener("mousemove", parallax);

const enter = (event: KeyboardEvent) => {
  if (event.code == "Enter") {
    console.log("pressed enter");
    checkReady();
    clear();
    (document.getElementById("title") as HTMLDivElement).style.display = "none";
    fetch(
      "http://4300showcase.infosci.cornell.edu:4522/search?" +
        // "http://localhost:5000/search?" +
        new URLSearchParams({
          title: (document.getElementById(
            "filter-text-val"
          ) as HTMLInputElement).value,
        }).toString()
    )
      .then((response) => response.json())
      .then((data) =>
        data.forEach((row: string[], i: number) => {
          // each row is [song name, song artist, song uri]
          result[i] = row;
          let tempDiv = document.createElement("div");
          tempDiv.setAttribute("data-id", i.toString());
          tempDiv.id = i.toString();
          tempDiv.onclick = function(e) {
            const element = e.target as HTMLElement;
            if (
              (document.getElementById(selected.toString()) as HTMLDivElement)
                .classList != null
            ) {
              (document.getElementById(
                selected.toString()
              ) as HTMLDivElement).classList.remove("selected");
            }
            element.classList.add("selected");
            const i = parseInt(element.getAttribute("data-id") || "0");
            selected = i;
            EmbedController.loadUri(result[i][2]);
          };
          tempDiv.innerHTML = songTemplate(row);
          const doc = document.getElementById("left") as HTMLElement;
          doc.appendChild(tempDiv);
        })
      )
      .then(() => {
        if (IFrameAPI == null) {
          // MAKE IT WAIT
        }
        // console.log(result);
        const element = document.getElementById("embed-iframe");
        const options = {
          width: "400",
          height: "400",
          uri: result[0][2],
        };
        const callback = (controller: any) => {
          EmbedController = controller;
        };
        IFrameAPI.createController(element, options, callback);
      });
  }
};

document.addEventListener("keydown", enter);

// const handlePress: KeyboardEventHandler<HTMLImageElement> = (e) => {
//   console.log("handler");
//   if (e.key === "Enter") {
//     console.log("enter");
//     search;
//   }
// };

window.onSpotifyIframeApiReady = (IFrameApi: any) => {
  IFrameAPI = IFrameApi;
  ready = true;
  console.log("ready");
};

function checkReady() {
  if (!ready) {
    window.setTimeout(checkReady, 50);
  }
}

const rocUp: MouseEventHandler<HTMLDivElement> = (e) => {
  if (
    !rel_track_list.includes(result[selected]) &&
    !irrel_track_list.includes(result[selected])
  ) {
    rel_track_list.push(result[selected]);
  }
  // (document.getElementById("thanks") as HTMLElement).classList.add("fade");
  // (document.getElementById("thanks") as HTMLElement).classList.remove("fade");
};

const rocDown: MouseEventHandler<HTMLDivElement> = (e) => {
  if (
    !rel_track_list.includes(result[selected]) &&
    !irrel_track_list.includes(result[selected])
  ) {
    irrel_track_list.push(result[selected]);
  }
  // (document.getElementById("thanks") as HTMLElement).classList.add("fade");
  // (document.getElementById("thanks") as HTMLElement).classList.remove("fade");
};

const regen: MouseEventHandler<HTMLDivElement> = (e) => {
  let send = {
    rel_track_list: rel_track_list,
    irrel_track_list: irrel_track_list,
  };
  clear();
  fetch("http://4300showcase.infosci.cornell.edu:4522/rocchio", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(send),
  })
    .then((response) => response.json())
    .then((data) => {
      data.forEach((row: string[], i: number) => {
        // each row is [song name, song artist, song uri]
        result[i] = row;
        let tempDiv = document.createElement("div");
        tempDiv.setAttribute("data-id", i.toString());
        tempDiv.onclick = function(e) {
          const element = e.target as HTMLElement;
          const i = parseInt(element.getAttribute("data-id") || "0");
          selected = i;
          EmbedController.loadUri(result[i][2]);
        };
        tempDiv.innerHTML = songTemplate(row);
        const doc = document.getElementById("left") as HTMLElement;
        doc.appendChild(tempDiv);
      });
    })
    .then(() => {
      if (IFrameAPI == null) {
        // MAKE IT WAIT
      }
      const element = document.getElementById("embed-iframe");
      const options = {
        width: "400",
        height: "400",
        uri: result[0][2],
      };
      const callback = (controller: any) => {
        EmbedController = controller;
      };
      IFrameAPI.createController(element, options, callback);
    });
};

export default App;
