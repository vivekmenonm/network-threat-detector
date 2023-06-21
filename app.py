import streamlit as st
from predict_traffic import predict_traffic_class
import numpy as np
# Streamlit app
def main():
    # Title and description
     # Title and description
    st.set_page_config(page_title="Malicious Network Traffic Detection", page_icon='🖥️')
    st.title("Threat Sense")
    st.write("Enter the values for the features and click 'Predict' to classify the nature of traffic.")

    # Feature input form
    st.subheader("Input Features")
    feature_names = ['tcp_packets', 'dist_port_tcp', 'external_ips', 'vulume_bytes', 'udp_packets',
                     'tcp_urg_packet', 'source_app_packets', 'remote_app_packets', 'source_app_bytes',
                     'remote_app_bytes', 'source_app_packets_2', 'dns_query_times']
    feature_values = []

    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0)
        feature_values.append(value)

    # Predict button
    if st.button("Predict"):
        # Prepare the features for prediction
        input_features = np.array([feature_values])

        # Predict the traffic class
        predicted_class = predict_traffic_class(input_features)

        # Display the predicted class
        st.subheader("Prediction Result")
        st.write(f"The network traffic class is: {predicted_class[0]}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
