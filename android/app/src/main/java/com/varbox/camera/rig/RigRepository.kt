package com.varbox.camera.rig

import com.google.firebase.auth.ktx.auth
import com.google.firebase.firestore.DocumentSnapshot
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/**
 * Rigs live in Firestore with Firebase Anonymous Auth. A rig holds its members, the chosen
 * settings and the director's record command; `android/firestore.rules` lets only member devices
 * read or write it.
 */
class RigRepository {
    private val auth = Firebase.auth
    private val db = Firebase.firestore

    /** Sign in anonymously once; the uid is stable for this install. */
    suspend fun signIn(): String {
        val existing = auth.currentUser
        if (existing != null) return existing.uid
        return auth.signInAnonymously().await().user!!.uid
    }

    fun currentUid(): String? = auth.currentUser?.uid

    private fun rigDoc(rigId: String) = db.collection("rigs").document(rigId)

    suspend fun createRig(name: String, uid: String, deviceName: String, model: String): String {
        val doc = db.collection("rigs").document()
        doc.set(
            mapOf(
                "name" to name,
                "directorId" to uid,
                "memberIds" to listOf(uid),
                "settings" to RigSettings().toMap(),
                "command" to RigCommand().toMap(),
                "createdAt" to FieldValue.serverTimestamp(),
            )
        ).await()
        doc.collection("members").document(uid).set(
            memberMap(uid, deviceName, "A", model)
        ).await()
        return doc.id
    }

    /** Add this device to a rig and give it the next free camera letter. */
    suspend fun joinRig(rigId: String, uid: String, deviceName: String, model: String): String {
        val doc = rigDoc(rigId)
        val letter = db.runTransaction { tx ->
            val snap = tx.get(doc)
            @Suppress("UNCHECKED_CAST")
            val members = (snap.get("memberIds") as? List<String>) ?: emptyList()
            if (!members.contains(uid)) {
                tx.update(doc, "memberIds", FieldValue.arrayUnion(uid))
            }
            val index = if (members.contains(uid)) members.indexOf(uid) else members.size
            ('A' + index).toString()
        }.await()
        doc.collection("members").document(uid).set(memberMap(uid, deviceName, letter, model)).await()
        return letter
    }

    private fun memberMap(uid: String, name: String, letter: String, model: String) = mapOf(
        "uid" to uid,
        "name" to name,
        "letter" to letter,
        "model" to model,
        "joinedAt" to FieldValue.serverTimestamp(),
    )

    fun observeRig(rigId: String): Flow<Rig?> = callbackFlow {
        val reg = rigDoc(rigId).addSnapshotListener { snap, _ ->
            trySend(snap?.let { parseRig(it) })
        }
        awaitClose { reg.remove() }
    }

    fun observeMembers(rigId: String): Flow<List<Member>> = callbackFlow {
        val reg = rigDoc(rigId).collection("members").addSnapshotListener { snap, _ ->
            val members = snap?.documents?.map { parseMember(it) } ?: emptyList()
            trySend(members.sortedBy { it.letter })
        }
        awaitClose { reg.remove() }
    }

    suspend fun setSettings(rigId: String, settings: RigSettings) {
        rigDoc(rigId).update("settings", settings.toMap()).await()
    }

    suspend fun sendCommand(rigId: String, command: RigCommand) {
        rigDoc(rigId).update("command", command.toMap()).await()
    }

    suspend fun publishCapabilities(rigId: String, uid: String, caps: Map<String, Any?>, model: String) {
        rigDoc(rigId).collection("members").document(uid)
            .set(mapOf("capabilities" to caps, "model" to model), com.google.firebase.firestore.SetOptions.merge())
            .await()
    }

    suspend fun publishApplied(rigId: String, uid: String, applied: Map<String, Any?>) {
        rigDoc(rigId).collection("members").document(uid)
            .set(mapOf("applied" to applied), com.google.firebase.firestore.SetOptions.merge()).await()
    }

    suspend fun publishClock(rigId: String, uid: String, clock: MemberClock) {
        rigDoc(rigId).collection("members").document(uid)
            .set(mapOf("clock" to clock.toMap()), com.google.firebase.firestore.SetOptions.merge()).await()
    }

    private fun parseRig(snap: DocumentSnapshot): Rig? {
        if (!snap.exists()) return null
        @Suppress("UNCHECKED_CAST")
        val members = (snap.get("memberIds") as? List<String>) ?: emptyList()
        @Suppress("UNCHECKED_CAST")
        val settings = snap.get("settings") as? Map<String, Any?>
        @Suppress("UNCHECKED_CAST")
        val command = snap.get("command") as? Map<String, Any?>
        return Rig(
            id = snap.id,
            name = snap.getString("name") ?: "",
            directorId = snap.getString("directorId") ?: "",
            memberIds = members,
            settings = RigSettings.fromMap(settings),
            command = RigCommand.fromMap(command),
        )
    }

    private fun parseMember(snap: DocumentSnapshot): Member {
        @Suppress("UNCHECKED_CAST")
        val caps = (snap.get("capabilities") as? Map<String, Any?>) ?: emptyMap()
        @Suppress("UNCHECKED_CAST")
        val applied = (snap.get("applied") as? Map<String, Any?>) ?: emptyMap()
        @Suppress("UNCHECKED_CAST")
        val clock = snap.get("clock") as? Map<String, Any?>
        return Member(
            uid = snap.getString("uid") ?: snap.id,
            name = snap.getString("name") ?: "",
            letter = snap.getString("letter") ?: "A",
            model = snap.getString("model") ?: "",
            capabilities = caps,
            applied = applied,
            clock = MemberClock.fromMap(clock),
        )
    }
}
