# Google sign-in

People sign in to varbox.guestpassvms.com with their Google account. The login page leads with a
"Sign in with Google" button (Google Identity Services, no redirect URI); the callback posts the ID
token to `POST /api/login/google`, which verifies it against Google's tokeninfo endpoint, checks the
audience, issuer, expiry and verified email, and signs in only accounts already on the team. An
unknown account is told to ask Shem. The owner keeps a password as a fallback behind a quiet "Sign
in with a password" link; no other account can sign in with a password.

The Team page now carries a name and a Google email per person, with no password, and the owner can
set or change any member's Google email — which is how existing members and Shem's own row get
linked. A member's first Google sign-in stores their account id and shows a "Google" tag beside
them. A `/privacy` page, linked from the landing and login pages, states what the site stores and
the contact address, which Google requires before the sign-in screen can be published.

## Steps left after deploy

1. Shem signs in with his password (the "Sign in with a password" link) and, on the Team page, sets
   his own Google email with **Set email**. From then on his Google account signs him in.
2. Shem sets each teammate's Google email on the Team page, so their Google sign-in matches.
3. In the Google Auth Platform **Branding** page for project `varbox-fdabe`, Shem adds the homepage
   link `https://varbox.guestpassvms.com` and the privacy link `https://varbox.guestpassvms.com/privacy`,
   then publishes the sign-in screen (moves it from Testing to In production).
