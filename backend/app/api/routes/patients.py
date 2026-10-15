"""Authenticated patient and patient-record APIs for the academic prototype."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, require_roles
from backend.app.db.models import Patient, PatientRecord
from backend.app.schemas.patient import PatientCreate, PatientListResponse, PatientResponse
from backend.app.schemas.patient_record import PatientRecordCreate, PatientRecordResponse


router = APIRouter(
    prefix="/patients",
    tags=["patients"],
    dependencies=[Depends(require_roles("admin", "clinician"))],
)


def _require_patient(database: Session, patient_id: int) -> Patient:
    patient = database.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return patient


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate, database: Session = Depends(get_db)) -> Patient:
    """Create a patient using only a unique project-level patient code."""
    existing = database.scalar(
        select(Patient.id).where(Patient.patient_code == payload.patient_code)
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="patient_code already exists")

    patient = Patient(patient_code=payload.patient_code)
    database.add(patient)
    try:
        database.commit()
        database.refresh(patient)
    except IntegrityError as exc:
        database.rollback()
        # Covers a duplicate inserted concurrently after the preflight query.
        if database.scalar(select(Patient.id).where(Patient.patient_code == payload.patient_code)):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="patient_code already exists",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Patient could not be created due to a data conflict",
        ) from exc
    return patient


@router.get("", response_model=PatientListResponse)
def list_patients(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    database: Session = Depends(get_db),
) -> PatientListResponse:
    """List patients with bounded pagination and minimal returned fields."""
    total = int(database.scalar(select(func.count()).select_from(Patient)) or 0)
    patients = database.scalars(
        select(Patient).order_by(Patient.id).offset(skip).limit(limit)
    ).all()
    return PatientListResponse(items=patients, skip=skip, limit=limit, total=total)


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: int, database: Session = Depends(get_db)) -> Patient:
    """Retrieve one patient by its database identifier."""
    return _require_patient(database, patient_id)


@router.post(
    "/{patient_id}/records",
    response_model=PatientRecordResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_patient_record(
    patient_id: int,
    payload: PatientRecordCreate,
    database: Session = Depends(get_db),
) -> PatientRecord:
    """Store submitted PIMA features as entered; no preprocessing is applied here."""
    _require_patient(database, patient_id)
    record = PatientRecord(patient_id=patient_id, **payload.model_dump())
    database.add(record)
    try:
        database.commit()
        database.refresh(record)
    except IntegrityError as exc:
        database.rollback()
        if database.get(Patient, patient_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found") from exc
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Patient record could not be created due to a data conflict",
        ) from exc
    return record


@router.get("/{patient_id}/records", response_model=list[PatientRecordResponse])
def list_patient_records(
    patient_id: int,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    database: Session = Depends(get_db),
) -> list[PatientRecord]:
    """List a patient's records in creation order with bounded pagination."""
    _require_patient(database, patient_id)
    return list(database.scalars(
        select(PatientRecord)
        .where(PatientRecord.patient_id == patient_id)
        .order_by(PatientRecord.created_at, PatientRecord.id)
        .offset(skip)
        .limit(limit)
    ).all())


@router.get(
    "/{patient_id}/records/{record_id}",
    response_model=PatientRecordResponse,
)
def get_patient_record(
    patient_id: int,
    record_id: int,
    database: Session = Depends(get_db),
) -> PatientRecord:
    """Retrieve a record only when it belongs to the patient in the URL."""
    _require_patient(database, patient_id)
    record = database.scalar(
        select(PatientRecord).where(
            PatientRecord.id == record_id,
            PatientRecord.patient_id == patient_id,
        )
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found")
    return record
