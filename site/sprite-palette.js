/* Decode the existing RGB material-mask frames; original artwork stays intact. */
globalThis.CREW_PALETTE = Object.freeze({
  recolor(rgba, color) {
    const palettes = {
      red: [[198, 17, 17], [148, 201, 219], [122, 8, 56]],
      black: [[63, 71, 78], [148, 201, 219], [30, 31, 38]]
    };
    if (!palettes[color]) throw new Error(`Unsupported material palette: ${color}`);
    const result = new Uint8ClampedArray(rgba);
    for (let i = 0; i < rgba.length; i += 4) {
      const channels = [rgba[i], rgba[i + 1], rgba[i + 2]];
      for (let dominant = 0; dominant < 3; dominant++) {
        if (channels[dominant] > 1.6 * channels[(dominant + 1) % 3] &&
            channels[dominant] > 1.6 * channels[(dominant + 2) % 3]) {
          for (let c = 0; c < 3; c++) result[i + c] = Math.round(channels[dominant] / 255 * palettes[color][dominant][c]);
          break;
        }
      }
    }
    return result;
  }
});
