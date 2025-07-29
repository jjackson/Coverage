SQL_QUERIES = {
    "opportunity_uservisit": "SELECT opportunity_uservisit.id as visit_id, opportunity_opportunity.name as opportunity_name, user_id AS flw_id, users_user.name AS flw_name, visit_date, opportunity_id, status, form_json -> 'form' ->> 'du_name' AS du_name, SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',1) AS lattitude, SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',2) AS longitude, SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',3) AS elevation_in_m, SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',4) AS accuracy_in_m, flagged, flag_reason, form_json -> 'form' -> 'cluster_update_block' -> 'update_cluster_case' -> 'case' ->> '@user_id' AS cchq_user_owner_id\n  FROM public.opportunity_uservisit \n  LEFT JOIN opportunity_opportunity ON opportunity_opportunity.id = opportunity_uservisit.opportunity_id  LEFT JOIN users_user ON users_user.id = opportunity_uservisit.user_id  "
                             "WHERE opportunity_opportunity.id IN (516,517,531,539,566,575,601,603);",

    "kmc_visit_query": """SELECT 
    opportunity_uservisit.id as visit_id,
    opportunity_opportunity.name as opportunity_name,
    user_id AS flw_id,
    users_user.name AS flw_name,
    visit_date,
    opportunity_id,
    status,
    SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',1) AS latitude,
    SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',2) AS longitude,
    SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',3) AS elevation_in_m,
    SPLIT_PART(form_json -> 'metadata' ->> 'location',' ',4) AS accuracy_in_m,
    flagged,
    flag_reason,
    form_json -> 'form' -> 'cluster_update_block' -> 'update_cluster_case' -> 'case' ->> '@user_id' AS cchq_user_owner_id,
    
    -- Case ID (updated to use the working field):
    form_json -> 'form' ->> 'kmc_beneficiary_case_id' AS "case_id",
    form_json -> 'form'->'anthropometric'->>'child_weight_visit' as "child_weight_visit",
    form_json
FROM public.opportunity_uservisit 
LEFT JOIN opportunity_opportunity ON opportunity_opportunity.id = opportunity_uservisit.opportunity_id
LEFT JOIN users_user ON users_user.id = opportunity_uservisit.user_id
WHERE opportunity_opportunity.id IN (524)""",

"flw_data_quality_analysis_query" : """SELECT opportunity_opportunity.name AS opportunity_name,
       opportunity_uservisit.id AS visit_id,
       opportunity_uservisit.user_id AS flw_id,
       users_user.name AS flw_name,
       opportunity_uservisit.visit_date,
       opportunity_uservisit.status,
       users_user.username,
       opportunity_deliverunit.name AS unit_name, -- Core demographics you requested
 form_json -> 'form' -> 'additional_case_info' ->> 'childs_age_in_month' AS child_age_months,
                                                   form_json -> 'form' -> 'additional_case_info' ->> 'childs_gender' AS child_gender,
                                                                                                     form_json -> 'form' -> 'additional_case_info' ->> 'household_phone' AS phone_number, -- MUAC (malnutrition screening) - all related fields
 form_json -> 'form' -> 'case' -> 'update' ->> 'MUAC_consent' AS muac_consent,
                                               form_json -> 'form' -> 'case' -> 'update' ->> 'muac_colour' AS muac_color,
                                                                                             form_json -> 'form' -> 'case' -> 'update' ->> 'soliciter_muac_cm' AS muac_measurement_cm,
                                                                                                                                           form_json -> 'form' -> 'muac_group' -> 'Muac' -> 'vitals' ->> 'muac_colour' AS muac_vitals_color, -- These seemed important for health context
 form_json -> 'form' ->> 'va_child_unwell_today' AS child_unwell_today,
                         form_json -> 'form' -> 'case' -> 'update' ->> 'diagnosed_with_mal_past_3_months' AS malnutrition_diagnosed_recently,
                                                                       form_json -> 'form' -> 'case' -> 'update' ->> 'under_treatment_for_mal' AS under_malnutrition_treatment, -- NEW DILIGENCE FIELDS
 form_json -> 'form' ->> 'received_va_dose_before' AS diligence_received_va_dose_before,
                         form_json -> 'form' ->> 'va_child_unwell_today' AS diligence_va_child_unwell_today,
                                                 form_json -> 'form' -> 'ors_group' ->> 'did_the_child_recover' AS diligence_ors_group_did_the_child_recover,
                                                                                        form_json -> 'form' -> 'case' -> 'update' ->> 'did_the_child_recover' AS diligence_did_the_child_recover,
                                                                                                                                      form_json -> 'form' ->> 'have_glasses' AS diligence_have_glasses,
                                                                                                                                                              form_json -> 'form' -> 'pictures' ->> 'received_any_vaccine' AS diligence_pictures_received_any_vaccine,
                                                                                                                                                                                                    form_json -> 'form' -> 'immunization_photo_group' ->> 'immunization_no_capture_reason' AS diligence_immunization_photo_group_immunization_no_capture_reason,
                                                                                                                                                                                                                                                          form_json -> 'form' ->> 'va_confirm_shared_knowledge' AS diligence_va_confirm_shared_knowledge,
                                                                                                                                                                                                                                                                                  form_json -> 'form' -> 'additional_case_info' ->> 'hh_have_children' AS diligence_hh_have_children,
                                                                                                                                                                                                                                                                                                                                    form_json -> 'form' -> 'pictures' ->> 'vaccine_not_provided_reason' AS diligence_pictures_vaccine_not_provided_reason,
                                                                                                                                                                                                                                                                                                                                                                          form_json -> 'form' ->> 'recent_va_dose' AS diligence_recent_va_dose,
                                                                                                                                                                                                                                                                                                                                                                                                  form_json -> 'form' -> 'case' -> 'update' ->> 'MUAC_consent' AS diligence_case_update_MUAC_consent,
                                                                                                                                                                                                                                                                                                                                                                                                                                                form_json -> 'form' -> 'ors_group' ->> 'still_facing_symptoms' AS diligence_ors_group_still_facing_symptoms,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       form_json -> 'form' -> 'case' -> 'update' ->> 'still_facing_symptoms' AS diligence_case_update_still_facing_symptoms,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     form_json -> 'form' ->> 'va_consent' AS diligence_va_consent
FROM opportunity_uservisit
LEFT JOIN opportunity_opportunity ON opportunity_opportunity.id = opportunity_uservisit.opportunity_id
LEFT JOIN users_user ON opportunity_uservisit.user_id = users_user.id
LEFT JOIN opportunity_deliverunit ON opportunity_uservisit.deliver_unit_id = opportunity_deliverunit.id
WHERE opportunity_opportunity.name IN ('ZEGCAWIS | CHC Givewell Scale Up')
ORDER BY opportunity_uservisit.visit_date;""", 


"sql_fetch_average_time_form_submission_last_7_days" :"""SELECT
  uv.user_id AS flw_id,
  uv.form_json -> 'metadata'->>'userID' AS cchq_user_id,
  ROUND(AVG(
    EXTRACT(EPOCH FROM (
      (uv.form_json -> 'metadata' ->> 'timeEnd')::timestamp - 
      (uv.form_json -> 'metadata' ->> 'timeStart')::timestamp
    )) / 60
  ), 2) AS avg_duration_minutes
FROM opportunity_uservisit uv
LEFT JOIN opportunity_opportunity oo 
    ON oo.id = uv.opportunity_id
LEFT JOIN users_user u 
    ON uv.user_id = u.id
WHERE 
  oo.name LIKE '%Scale Up%'
  AND uv.visit_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY uv.user_id, u.name, oo.name, cchq_user_id""", 


"opp_user_details_mapping" : """SELECT
  oo.name AS opportunity_name,
  uv.user_id AS flw_id,
  u.name AS flw_name,
  u.username,
  uv.form_json -> 'metadata'->>'userID' AS cchq_user_id
FROM opportunity_uservisit uv
LEFT JOIN opportunity_opportunity oo 
    ON oo.id = uv.opportunity_id
LEFT JOIN users_user u 
    ON uv.user_id = u.id
WHERE 
  oo.name LIKE '%Scale Up%'
GROUP BY uv.user_id, u.name, oo.name, cchq_user_id,u.username
ORDER BY opportunity_name, flw_name;"""
}
