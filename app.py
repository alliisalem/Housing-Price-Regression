import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="California Housing Price Predictor", layout="centered")


MODEL_PATH = "models/best_model.pkl"


def load_model():
    """Load the trained model saved from Phase 4."""
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except FileNotFoundError:
        return None, f"Model file not found at: {MODEL_PATH}"
    except Exception as e:
        return None, f"Error loading model: {e}"


def build_input_df(
    median_income: float,
    house_age: float,
    average_rooms: float,
    average_bedrooms: float,
    population: float,
    average_occupancy: float,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """
    Build an input row using the same feature names used in the project.
    The app defaults to the renamed columns from Phase 3.
    """
    data = {
        "MedianIncome": [median_income],
        "HouseAge": [house_age],
        "AverageRooms": [average_rooms],
        "AverageBedrooms": [average_bedrooms],
        "Population": [population],
        "AverageOccupancy": [average_occupancy],
        "Latitude": [latitude],
        "Longitude": [longitude],
    }

    # Derived features created in Phase 3
    rooms_per_occupant = average_rooms / average_occupancy if average_occupancy != 0 else 0.0
    bedroom_room_ratio = average_bedrooms / average_rooms if average_rooms != 0 else 0.0
    population_per_room = population / average_rooms if average_rooms != 0 else 0.0

    data["RoomsPerOccupant"] = [rooms_per_occupant]
    data["BedroomRoomRatio"] = [bedroom_room_ratio]
    data["PopulationPerRoom"] = [population_per_room]

    return pd.DataFrame(data)

def align_features_to_model(model, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder / rename features to match the fitted model if feature names are available.
    This supports either:
    - original California Housing names
    - renamed Phase 3 names
    """
    if not hasattr(model, "feature_names_in_"):
        return input_df

    expected = list(model.feature_names_in_)

    original_to_phase3 = {
        "MedInc": "MedianIncome",
        "AveRooms": "AverageRooms",
        "AveBedrms": "AverageBedrooms",
        "AveOccup": "AverageOccupancy",
        "MedHouseVal": "MedianHouseValue",
    }

    phase3_to_original = {v: k for k, v in original_to_phase3.items()}

    aligned = input_df.copy()

    # If model expects original sklearn names, rename Phase 3 columns back
    if any(col in expected for col in phase3_to_original.values()):
        aligned = aligned.rename(columns=phase3_to_original)

    # If model expects Phase 3 renamed columns, keep them as they are
    elif any(col in expected for col in original_to_phase3.values()):
        pass

    missing_cols = [col for col in expected if col not in aligned.columns]
    if missing_cols:
        raise ValueError(
            "The saved model expects features that are missing from the app input: "
            + ", ".join(missing_cols)
        )

    return aligned[expected]

st.title("California Housing Price Predictor")
st.write(
    "Enter housing district features below to predict the median house value "
    "using your trained machine learning model."
)

model, load_error = load_model()

if load_error:
    st.error(load_error)
    st.info(
        "Make sure you saved your best model from Phase 4 using:\n\n"
        "joblib.dump(best_model, 'models/best_model.pkl')"
    )
    st.stop()

with st.form("prediction_form"):
    st.subheader("Input Features")

    median_income = st.number_input("Median Income", min_value=0.0, value=3.87, step=0.01)
    house_age = st.number_input("House Age", min_value=0.0, value=28.0, step=1.0)
    average_rooms = st.number_input("Average Rooms", min_value=0.1, value=5.43, step=0.01)
    average_bedrooms = st.number_input("Average Bedrooms", min_value=0.1, value=1.10, step=0.01)
    population = st.number_input("Population", min_value=1.0, value=1425.0, step=1.0)
    average_occupancy = st.number_input("Average Occupancy", min_value=0.1, value=3.07, step=0.01)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.26, step=0.01)
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.49, step=0.01)

    actual_value = st.number_input(
        "Actual House Value (optional, for undervaluation check)",
        min_value=0.0,
        value=0.0,
        step=0.01,
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_df = build_input_df(
            median_income=median_income,
            house_age=house_age,
            average_rooms=average_rooms,
            average_bedrooms=average_bedrooms,
            population=population,
            average_occupancy=average_occupancy,
            latitude=latitude,
            longitude=longitude,
        )

        model_input = align_features_to_model(model, input_df)
        prediction = model.predict(model_input)[0]

        st.success(f"Predicted Median House Value: {prediction:.4f}")

        st.subheader("Input Summary")
        st.dataframe(input_df, use_container_width=True)

        if actual_value > 0:
            difference = prediction - actual_value

            st.subheader("Undervaluation Check")
            st.write(f"Predicted Value: {prediction:.4f}")
            st.write(f"Actual Value: {actual_value:.4f}")
            st.write(f"Difference (Predicted - Actual): {difference:.4f}")

            if difference > 0:
                st.info(
                    "This district may be **undervalued** because the predicted value "
                    "is higher than the actual value."
                )
            elif difference < 0:
                st.warning(
                    "This district may be **overpriced** because the predicted value "
                    "is lower than the actual value."
                )
            else:
                st.write("The predicted and actual values are equal.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
