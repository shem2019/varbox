# Web camera

Any phone records for the VAR Box rig from a link, with nothing to install. The page at
`/camera` stamps every frame with its capture time on a clock shared by all phones, so the
cameras line up by clock alone, however loud the arena is.

## How it works

- **Joining:** Team → Add a phone shows a QR of `https://varbox.guestpassvms.com/camera#<token>`.
  Scanning it with the phone's camera opens the page signed in as that phone. The token sits in
  the link's fragment, which never reaches the server, and the page removes it from the address
  bar.
- **Shared clock:** the phone exchanges 40 timestamps with `/api/clock` and keeps the fastest
  quarter: the offset is their median and the uncertainty half the fastest round trip. It
  re-measures every 10 s, and the sidecar keeps every measurement so drift is corrected per frame.
  The page shows "Synced ±N ms" in green when the uncertainty is within half a frame.
- **Recording:** a worker takes each camera frame (`MediaStreamTrackProcessor`), encodes it
  itself (WebCodecs, H.264) and writes a fragmented MP4 to the phone's own storage as it records.
  The MP4 timestamps are the frames' capture times. The sidecar keeps the native app's format and
  records which clock the frame timestamps were on (`frame_times_source`), found by checking that
  every frame lands 0–250 ms before it reached the page.
- **Uploading:** after a take, the page sends the sidecar and then the video, 8 MB at a time.
  After a network drop or a reload it asks the server how much has arrived and carries on. A take
  is deleted from the phone once the server holds all of it.
- **Leaving the page:** if the phone is locked or the page is hidden mid-take, the file is closed
  cleanly and the sidecar marked `interrupted`.

Chrome on Android has every part. A browser without `MediaStreamTrackProcessor` (Safari) stamps
frames as they reach the page through `requestVideoFrameCallback`. Its sync quality is what the
test below establishes.

## Sync test

The test measures how closely two phones' stamps agree, with light rather than sound.

1. Deploy, then on a laptop open https://varbox.guestpassvms.com/sync-flash full screen. It draws
   the shared clock as a code that changes 60 times a second.
2. Add two phones on the Team page and open each camera link. Give both the same **Take** name
   (for example `sync-wifi`), set cameras A and B, and choose the frame rate to test.
3. Point both phones at the laptop screen so the code fills most of each frame. Press Record on
   both, film for 30 seconds, then press Stop. Both takes upload on their own.
4. Run the tool with the site's worker token:

   ```bash
   export VARBOX_WORKER_TOKEN=<worker_token from storage/config.json>
   .venv/bin/python -m boxing_analytics.mesh4d.syncflash --server https://varbox.guestpassvms.com --take sync-wifi
   ```

5. Repeat with one phone on mobile data (take `sync-mobile`), and with an iPhone if one is to be
   used.

The tool prints PASS when the median difference between the two phones is within one frame and
95% of frame pairs are within two, and writes the details to `sync_test.json`:
- `median_ms` is the clock error between the phones. Each pair of frames that saw the same code was
  captured within one display tick (17 ms) of each other, so single pairs scatter and the median
  over hundreds of pairs is the result.
- `frame_times_source` says which route each phone's browser used.
- `clock_uncertainty_ms` is each phone's own estimate.

Each test take also creates a pending task, since it has two cameras; delete it from Pending
tasks.

A browser-only run on 19 November recorded two canvas streams of the flash code through this
whole path (recorder, phone storage, upload, tool). Both files decoded 180 of 180 frames,
matching their sidecars, and the two agreed to −0.2 ms. That run checks the software; the phone
test measures the cameras.
