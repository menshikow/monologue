const worker = new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
});

worker.onmessage = (e) => {
    console.log(e.data);
};

worker.postMessage("ping");
