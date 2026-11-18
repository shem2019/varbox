# Phone camera rig

Two Android phones stand in for dedicated cameras. The VAR Box Camera app pairs the phones into one
rig, pushes identical capture settings to each, and starts recording on all of them at a single
shared instant. After the take, every phone uploads its video and a sidecar to
varbox.guestpassvms.com through its own upload token. When each camera of a recording has arrived,
the site creates a **pending task** and emails Shem; pressing **Process** on the dashboard queues
it, and the existing GPU worker picks it up.

The cameras line up by time two ways. Each phone measures its clock against a shared SNTP reference
and stamps every frame with its capture time on that clock, so the sidecars give an exact offset
between the cameras. The pipeline uses that offset and keeps the audio clap-and-whistle sync as a
cross-check, flagging any disagreement above one frame. Footage without sidecars still syncs from
audio alone.

## Remaining steps

Run these in order. The repository holds everything else.

### 1. Firebase

1. Create a project at https://console.firebase.google.com and add an Android app with package
   name `com.varbox.camera`.
2. Download `google-services.json` into `android/app/`:
   ```
   cp ~/Downloads/google-services.json android/app/google-services.json
   ```
3. In the console, enable **Firestore Database** (Native mode) and **Authentication → Anonymous**.
4. Deploy the rules with the Firebase CLI:
   ```
   npm install -g firebase-tools
   firebase login
   cd android
   printf '{\n  "firestore": { "rules": "firestore.rules" }\n}\n' > firebase.json
   firebase deploy --project <your-project-id> --only firestore:rules
   ```

### 2. Mail on the server

Add an `mail` block to the site's `storage/config.json` so the pending-task email sends. Without
it the message is written to `storage/mail.log` and the task is still created.

```
ssh <user>@<server>
# edit /www/wwwroot/varbox.guestpassvms.com/storage/config.json, add:
#   "mail": {
#     "host": "smtp.yourprovider.com",
#     "port": 587,
#     "user": "varbox@yourdomain.com",
#     "pass": "APP_PASSWORD",
#     "from": "varbox@yourdomain.com",
#     "to": "shem@yourdomain.com"
#   }
```

### 3. Install and pair two phones

```
cd android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

On each phone: open VAR Box Camera, and on the dashboard open **Team → Add a phone** and scan the QR
to sign the phone in. On one phone create a rig (it becomes the director); on the dashboard's **Rig**
tab it shows a join QR that the other phone scans. On the Rig tab press **Measure clock** on each
phone, then **Apply to every phone** as the director. Open the **Record** tab on both and press
**Start take** on the director.

### 4. Clap test

Record a short take with one loud clap near the start. After it processes, open the run's
`sync.json`: the `sidecar` and `audio` offsets should agree within one frame and `agree` should be
`true`, so the `sidecar` offset (anchored on each file's first frame) is used. The worker log prints
both offsets. If they differ by more than one frame, `agree` is `false` and the pipeline falls back
to the `audio` offset, which is measured on the files themselves.

### 5. Deploy the site

The web changes ship when `web/**` lands on `master`:

```
git checkout master
git merge feature/phone-rig
git push origin master
```

`deploy-web.yml` checks PHP syntax, uploads the release and reloads nginx.
