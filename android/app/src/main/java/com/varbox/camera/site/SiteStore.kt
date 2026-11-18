package com.varbox.camera.site

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.varbox.camera.SitePairing
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore("varbox")

/** Where the phone keeps its site sign-in (URL and upload token) and its rig membership. */
class SiteStore(private val context: Context) {
    private val url = stringPreferencesKey("site_url")
    private val token = stringPreferencesKey("site_token")
    private val phoneName = stringPreferencesKey("phone_name")
    private val rigId = stringPreferencesKey("rig_id")
    private val wifiOnly = booleanPreferencesKey("wifi_only")

    data class Site(val url: String, val token: String, val phoneName: String)

    val site: Flow<Site?> = context.dataStore.data.map { p ->
        val u = p[url]
        val t = p[token]
        if (u.isNullOrEmpty() || t.isNullOrEmpty()) null else Site(u, t, p[phoneName] ?: "Phone")
    }

    val rig: Flow<String?> = context.dataStore.data.map { it[rigId] }
    val wifiOnlyFlow: Flow<Boolean> = context.dataStore.data.map { it[wifiOnly] ?: false }

    suspend fun saveSite(pairing: SitePairing) {
        context.dataStore.edit {
            it[url] = pairing.url
            it[token] = pairing.token
            it[phoneName] = pairing.name
        }
    }

    suspend fun saveRig(id: String) {
        context.dataStore.edit { it[rigId] = id }
    }

    suspend fun setWifiOnly(value: Boolean) {
        context.dataStore.edit { it[wifiOnly] = value }
    }

    suspend fun current(): Site? = site.first()
    suspend fun currentRig(): String? = rig.first()
}
