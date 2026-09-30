# Vercel dashboard

This directory contains the Vercel-hosted static dashboard. The full Keras model and the interactive Streamlit app remain in the parent `outputs` directory; Vercel serves this dashboard and its saved metrics/graphs.

From this directory, refresh the dashboard payload after training and deploy:

```bash
python build_site.py
npx vercel --prod
```

Vercel hosts the static dashboard; it does not execute the local Streamlit server or retrain the LSTM. To update the site, retrain from the parent folder, then run the build and deployment commands above.

If the CLI is not logged in, `npx vercel deploy --temporary --yes` creates a temporary preview and a private claim link. Claim that deployment in your Vercel account to keep it live; temporary previews expire after 60 minutes.
