package com.varbox.camera

import org.json.JSONObject

/** Payload encoded in the dashboard's "Add a phone" QR. */
data class SitePairing(val url: String, val token: String, val name: String)

/** The two QR payloads the app scans: one signs the phone in to the site, one joins a rig. */
object Wire {
    fun parseSitePairing(text: String): SitePairing? = try {
        val o = JSONObject(text)
        val url = o.optString("url").trimEnd('/')
        val token = o.optString("token")
        if (o.optString("kind") == "varbox-phone" && url.isNotEmpty() && token.isNotEmpty()) {
            SitePairing(url, token, o.optString("name", "Phone"))
        } else {
            null
        }
    } catch (e: Exception) {
        null
    }

    fun rigJoinPayload(rigId: String, name: String): String =
        JSONObject().put("kind", "varbox-rig").put("rig", rigId).put("name", name).toString()

    fun parseRigJoin(text: String): String? = try {
        val o = JSONObject(text)
        if (o.optString("kind") == "varbox-rig") o.optString("rig").ifEmpty { null } else null
    } catch (e: Exception) {
        null
    }
}
